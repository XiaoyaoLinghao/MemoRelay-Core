"""V8: validate each page before caching; repair only that page, at most once.

Validation is repeated by the ordinary atomic whole-source pipeline. This layer
never writes claims, edits source text or promotes evidence authority.
"""
from __future__ import annotations

import re
from knowledge_weaver import contracts
from .proposals import ExtractionError, resolve_proposal
from .context import anchor_context, repair_feedback
from .echoes import split_assistant_echoes


def schema_details(issues):
    fields = set()
    def walk(node):
        if isinstance(node, dict):
            fields.update(node.get('properties', {}))
            for value in node.values(): walk(value)
        elif isinstance(node, list):
            for value in node: walk(value)
    walk(contracts.extraction_schema())
    result = []
    for issue in issues[:8]:
        tokens = re.findall(r'\.[A-Za-z_][A-Za-z_0-9-]*|\[\d{1,8}\]', issue.path)
        path = '$' + ''.join(token if token.startswith('[') or token[1:] in fields else '.*' for token in tokens)
        message = issue.message
        code = next((code for prefix,code in (
            ('missing required','missing_required'), ('unknown property','unknown_property'),
            ('expected type','wrong_type'), ('must be one of','enum'), ('must equal','constant'),
            ('at most','upper_bound'), ('at least','lower_bound'), ('must be <','upper_bound'), ('must be >','lower_bound'), ('match pattern','pattern'),
            ('anyOf','any_of'), ('oneOf','one_of')) if prefix in message), 'other_validation')
        item = {'path':path[:200], 'code':code}
        bound = re.fullmatch(r'(?:length )?must be (<=|>=|<|>) (\d{1,9})', message)
        if bound: item.update(comparison=bound[1], bound=int(bound[2]))
        expected = re.match(r"expected type (\[[^\]]{1,100}\]), got ", message)
        if expected:
            types = re.findall(r"'(object|array|string|integer|number|boolean|null)'", expected[1])
            if types: item['expected_types'] = types

        match = re.search(r"missing required property '([^']+)'", message)
        if match and match[1] in fields: item['required_field'] = match[1]
        result.append(item)
    return {'issues':result, 'total_issue_count':len(issues)}


def validation_error(reason, *, issues=None):
    error = ExtractionError('MODEL_UNAVAILABLE', 'page validation failed', retryable=True, category=reason)
    error.validation_details = schema_details(issues) if issues else {}
    return error


def evidence_audit(proposal, parts):
    """Only bounded indices and booleans, never evidence text or unknown keys."""
    by_key = {(p.event_id,p.event_revision,p.part_index):p for p in parts}
    invalid = []
    claims = proposal.get('claims') if isinstance(proposal,dict) else None
    for ci, claim in enumerate(claims[:100] if isinstance(claims,list) else []):
        refs = claim.get('evidence') if isinstance(claim,dict) else None
        for ri, ref in enumerate(refs if isinstance(refs,list) else []):
            if not isinstance(ref,dict): continue
            key = (ref.get('event_id'),ref.get('event_revision'),ref.get('part_index'))
            typed = isinstance(key[0],str) and type(key[1]) is int and type(key[2]) is int
            part = by_key.get(key) if typed else None
            quote = ref.get('support_quote')
            exact = bool(part is not None and isinstance(quote,str) and quote and quote in part.text)
            too_long = isinstance(quote,str) and len(quote)>2000
            if not exact or too_long:
                invalid.append({'claim_index':ci,'reference_index':ri,'source_exists':part is not None,
                                'exact_substring':exact,'over_quote_limit':too_long})
                if len(invalid) == 8: return invalid
    return invalid


def review_notes(values):
    """Ephemeral, untrusted critique; never persisted in failure metadata."""
    notes=[]; budget=2048
    for value in values[:8]:
        if not isinstance(value,str) or contracts.secret_kinds(value): continue
        encoded=value.encode('utf-8')[:min(512,budget)]
        note=encoded.decode('utf-8','ignore')
        if note: notes.append(note);budget-=len(note.encode('utf-8'))
        if budget<=0: break
    return notes


def secret_locations(proposal):
    found=[]
    for i, claim in enumerate(proposal['claims']):
        for field in ('subject_hint','qualifiers','value','display_text','evidence'):
            kinds=contracts.secret_kinds(contracts.compact_json(claim.get(field)))
            if kinds: found.append({'path':f'$.claims[{i}].{field}','kinds':sorted(kinds)})
            if len(found)>=8: return found
    if contracts.secret_kinds(contracts.compact_json(proposal['warnings'])):
        found.append({'path':'$.warnings','kinds':sorted(contracts.secret_kinds(contracts.compact_json(proposal['warnings'])))})
    return found[:8]


def resolution_issues(warnings):
    issues=[]
    for note in warnings:
        match=re.match(r'^claim\[(\d+)\]: (.*)$',note)
        if not match: continue
        message=match[2]
        code=None
        if 'target_claim_hint is not a compatible scoped field' in message:code='target_incompatible'
        elif message.startswith('unresolved project hint'):code='unknown_project'
        elif message.startswith('quarantined (proposal still contains secret content)'):code='secret_content'
        elif message.startswith('dropped (explicit correction requires native user evidence)'):code='correction_requires_native_user'
        elif message.startswith('dropped ('):code='invalid_evidence'
        if code:issues.append({'claim_index':int(match[1]),'code':code})
        if len(issues)>=8:break
    return issues


def previous_claim_fragments(proposal, feedback):
    """Supply the actual rejected fields to a stateless retry, bounded and secret-checked."""
    indices=sorted({x['claim_index'] for name in ('evidence_audit','resolver_issues')
                    for x in feedback.get(name,[]) if type(x.get('claim_index')) is int})[:8]
    claims=proposal.get('claims',[]) if isinstance(proposal,dict) else []
    def unsafe(value):
        if isinstance(value,dict):
            if any(str(k).lower() in {'password','passwd','api_key','apikey','secret','access_token','private_key','token','authorization','credential'} for k in value): return True
            return any(unsafe(v) for v in value.values())
        if isinstance(value,list):return any(unsafe(v) for v in value)
        return isinstance(value,str) and bool(contracts.secret_kinds(value))
    result=[];budget=8192
    for index in indices:
        if not 0<=index<len(claims) or not isinstance(claims[index],dict):continue
        fields={};omitted=[]
        for name in ('project_hint','subject_hint','predicate','facet','target_claim_hint','qualifiers','evidence'):
            if name not in claims[index]:continue
            value=claims[index][name];encoded=contracts.compact_json({name:value})
            if unsafe(value) or contracts.secret_kinds(encoded) or len(encoded.encode('utf-8'))>min(4096,budget-256):
                omitted.append(name);continue
            fields[name]=value;budget-=len(encoded.encode('utf-8'))
        if fields:
            entry={'claim_index':index,'fields':fields,'omitted_fields':omitted}
            if len(contracts.compact_json(result+[entry]).encode('utf-8'))>8192:break
            result.append(entry)
        if budget<256:break
    return result


class ValidatedPageExtractor:
    def __init__(self, store, inner, group, full_parts):
        self.store, self.inner, self.group, self.parts = store, inner, group, full_parts
        self.calls = 0
        self.cached_error = None

    def validate(self, proposal, parts, context):
        enhanced = context.get('prompt_version') in ('prompt:9', 'prompt:10')
        issues = contracts.validate_extraction_output(proposal)
        audit = evidence_audit(proposal,self.parts) if enhanced else []
        if issues:
            exc = validation_error('schema_invalid', issues=issues)
            if audit: exc.repair_context = {'evidence_audit':audit}
            raise exc
        if audit:
            exc = validation_error('evidence_invalid')
            exc.repair_context = {'evidence_audit':audit}
            raise exc
        if context.get('segment') and contracts.secret_kinds(contracts.compact_json(proposal)):
            exc = validation_error('secret_content')
            if enhanced: exc.repair_context = {'secret_locations':secret_locations(proposal)}
            raise exc
        if context.get('segment') and not proposal['claims'] and not {p.event_id for p in parts}.issubset(set(proposal['excluded_event_ids'])):
            raise validation_error('missing_disposition')
        claims, warnings = resolve_proposal(self.group, proposal, self.parts)
        rejected=resolution_issues(warnings) if enhanced else []
        if rejected:
            exc=validation_error('proposal_invalid')
            exc.repair_context={'resolver_issues':rejected}
            raise exc
        with self.store.read() as conn:
            fresh, _ = split_assistant_echoes(conn, claims)
        if proposal.get('warnings') and fresh:
            exc = ExtractionError('MODEL_REVIEW_REQUIRED', 'model proposal requires review', retryable=True, category='model_review_required')
            if enhanced: exc.repair_context = {'untrusted_review_notes':review_notes(proposal['warnings'])}
            raise exc
        return proposal

    def _feedback(self, context, exc):
        ctx = dict(context)
        category = getattr(exc, 'category', None)
        if exc.message.startswith('FIELD_IDENTITY_COLLISION:'):
            ctx.update(anchor_context(self.group, exc.message))
            ctx['previous_validation_error'] = repair_feedback(exc.message)
            category = 'field_identity_collision'
        ctx['page_validation_feedback'] = {
            'category':category or ('model_review_required' if exc.code=='MODEL_REVIEW_REQUIRED' else 'invalid_response'),
            **getattr(exc, 'validation_details', {}),
        }
        if ctx.get('prompt_version') in ('prompt:9', 'prompt:10'):
            ctx['page_validation_feedback'].update(getattr(exc,'repair_context',{}))
        return ctx

    def extract(self, parts, context):
        ctx = dict(context)
        if self.cached_error:
            ctx = self._feedback(ctx, self.cached_error)
            self.cached_error = None
        for attempt in range(2):
            if self.calls >= 64: raise validation_error('call_budget')
            self.calls += 1
            proposal = None
            try:
                proposal = self.inner.extract(parts, ctx)
                return self.validate(proposal, parts, ctx)
            except ExtractionError as exc:
                category = getattr(exc, 'category', None)
                eligible = category in {'schema_invalid','secret_content','missing_disposition','invalid_response','model_review_required','evidence_invalid','proposal_invalid'} or exc.code=='MODEL_REVIEW_REQUIRED' or exc.message.startswith('FIELD_IDENTITY_COLLISION:')
                if attempt or not eligible:
                    # Numeric location only: never the source, model output or exception text.
                    segment = ctx.get('segment') or {}
                    exc.segment_ranges = [{k:r[k] for k in ('part_index','start_byte','end_byte','window_index') if type(r.get(k)) is int} for r in segment.get('ranges',[])[:8]]
                    raise
                ctx = self._feedback(ctx, exc)
                if ctx.get('prompt_version') in ('prompt:9', 'prompt:10') and proposal is not None:
                    fragments = previous_claim_fragments(proposal,ctx['page_validation_feedback'])
                    if fragments: ctx['page_validation_feedback']['untrusted_previous_claims'] = fragments
