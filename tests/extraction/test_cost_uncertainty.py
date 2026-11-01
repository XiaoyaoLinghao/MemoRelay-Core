"""Consumed-but-timed-out model requests must not be reported as free."""
import json,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import pytest
from knowledge_weaver.extraction import HttpExtractor,run_worker_once
from knowledge_weaver.store import current_items
from .test_source_lifecycle import setup
from .test_extraction import _claim,_proposal

@pytest.mark.parametrize("recover",[True,False])
def test_unknown_model_charge_is_durable_bounded_and_success_is_not_recomputed(store,recover):
 event,subject,part=setup(store);proposal=_proposal(_claim(part,assertion_kind="explicit_statement"))
 received=[]
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def do_POST(self):
   self.rfile.read(int(self.headers["Content-Length"]));received.append(1)
   if not recover or len(received)==1:time.sleep(.3)
   body=json.dumps({"choices":[{"message":{"content":json.dumps(proposal)}}]}).encode()
   try:
    self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)
   except (BrokenPipeError,ConnectionResetError):pass
 server=ThreadingHTTPServer(("127.0.0.1",0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 model=HttpExtractor(f"http://127.0.0.1:{server.server_port}/chat/completions",model="synthetic-loopback",timeout=.08)
 try:
  now=None;states=[]
  for _ in range(3):
   result=run_worker_once(store,model,lease_seconds=240,now=now);states.append(result["state"])
   with store.read() as c:job=dict(c.execute("SELECT * FROM jobs WHERE kind='extract'").fetchone())
   if result["state"] in ("succeeded","failed"):break
   now=job["run_after"]
  with store.read() as c:attempts=[dict(x) for x in c.execute("SELECT * FROM job_attempts ORDER BY attempt")]
  uncertain=[x for x in attempts if x["error_code"]=="MODEL_UNAVAILABLE"]
  assert uncertain and all(json.loads(x["detail"])=={"billing_outcome":"unknown","provider_usage":None,"reason":"external_request_outcome_unknown"} for x in uncertain)
  assert received and len(received)==len(attempts)
  assert states==(["retry_wait","succeeded"] if recover else ["retry_wait","retry_wait","failed"])
  count=len(received);assert run_worker_once(store,model)["claimed"]==0 and len(received)==count
  if recover:assert any(x["value"]==4200 for x in current_items(store,workspace_id="owner:demo"))
  else:assert current_items(store,workspace_id="owner:demo")==[]
 finally:
  server.shutdown();server.server_close();thread.join(timeout=2)
