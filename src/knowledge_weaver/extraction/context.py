"""Bounded v7 field anchors and native source scope; neither is fresh evidence."""
from __future__ import annotations

import re
from knowledge_weaver import contracts


def repair_feedback(previous: str | None) -> str | None:
    match = re.match(r'(FIELD_IDENTITY_COLLISION: null target_claim_hint collides with claim:[a-f0-9]+)', previous or '')
    if not match:
        return previous
    return match[1] + (
        '. Determine the original source identity again; a previous null target is not authoritative. '
        'Use an existing target only for a genuinely identical entity/property/business scope. '
        'For an independent invocation result use its real source observation identity; '
        'for a stable property update use its matching anchor. Do not bind an unrelated field, '
        'invent business qualifiers, or discard supported facts merely to make validation pass.'
    )


def source_scope(group: dict) -> dict:
    aliases = group.get('source_aliases') or []
    tools = sorted({item[len('native-tool:'):] for item in aliases
                    if isinstance(item, str)
                    and re.fullmatch(r'native-tool:[A-Za-z0-9_]{1,64}', item)})
    return {'event_id': group['event_id'], 'origin_kind': group['origin_kind'],
            'native_tool': tools[0] if len(tools) == 1 else None,
            'observation_id': group['event_id']}


def anchor_context(group: dict, previous_error: str | None, *, budget: int = 8192) -> dict:
    """Keep the explicitly collided field in view without inventing a target.

    The backend still validates the model's independent identity decision.
    Oversized/secret anchors do not hide the subsequent smaller candidates.
    Both lists together share one byte budget, including their field names.
    """
    candidates = list(group.get('claim_candidates') or [])
    match = re.match(r'FIELD_IDENTITY_COLLISION: null target_claim_hint collides with (claim:[a-f0-9]+)',
                     previous_error or '')
    preferred = match[1] if match else None
    candidates.sort(key=lambda item: item['claim_id'] != preferred)
    result = {'known_claims': [], 'known_claim_details': []}
    for entry in candidates:
        basic = {key: entry.get(key) for key in
                 ('claim_id', 'project_id', 'predicate', 'facet', 'label', 'qualifiers')}
        detail = {key: entry.get(key) for key in
                  ('claim_id', 'subject_id', 'value', 'state', 'authority', 'ordering_time')}
        if contracts.secret_kinds(contracts.compact_json([basic, detail])):
            continue
        serialized_value = contracts.compact_json(detail['value'])
        if len(serialized_value.encode('utf-8')) > 1024:
            detail.pop('value')
            detail.update(value_digest=contracts.sha256_hex(serialized_value), value_omitted_large=True)
        proposed = {'known_claims': result['known_claims'] + [basic],
                    'known_claim_details': result['known_claim_details'] + [detail]}
        if len(contracts.compact_json(proposed).encode('utf-8')) <= budget:
            result = proposed
    return result
