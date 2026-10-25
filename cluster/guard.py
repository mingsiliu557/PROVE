"""Bound this test process tree; never terminate unrelated node processes."""
import ctypes,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import psutil
command=sys.argv[1:];assert command
limit=int(os.environ.get('TTRV_REQUESTED_MEMORY_MB',os.environ.get('SLURM_MEM_PER_NODE','4096')))*1024**2
report=Path(os.environ['TTRV_GUARD_REPORT']);deadline=time.monotonic()+int(os.environ.get('TTRV_GUARD_SECONDS','1500'))
# On Linux, adopt descendants that daemonize so they cannot escape accounting
# or cleanup when their immediate parent exits.
if sys.platform=='linux':
 if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:raise OSError(ctypes.get_errno(),'PR_SET_CHILD_SUBREAPER failed')
parent=psutil.Process()
p=subprocess.Popen(command,start_new_session=True);root=psutil.Process(p.pid);peak=0;reason=None;last_check=0;owned={};rc=1
def interrupted(sig,frame):
 global reason
 reason='guard_interrupted_'+str(sig)
 raise InterruptedError(reason)
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupted)
try:
 while p.poll() is None:
  total=0
  try:procs=parent.children(recursive=True)
  except psutil.NoSuchProcess:break
  for proc in procs:
   try:
    owned[proc.pid]=(proc,proc.create_time())
    info=proc.memory_full_info();total+=getattr(info,'pss',info.rss)
   except (psutil.NoSuchProcess,psutil.AccessDenied):pass
  peak=max(peak,total)
  if os.environ.get('CLUSTER_VERIFIED_JOB_ID') and time.monotonic()-last_check>15:
   try:
    status=subprocess.run(['squeue','-j',os.environ['CLUSTER_VERIFIED_JOB_ID'],'-h','-o','%T'],text=True,capture_output=True,timeout=10)
    if status.returncode or status.stdout.strip()!='RUNNING':reason='allocation_no_longer_running'
   except (OSError,subprocess.TimeoutExpired):reason='cannot_verify_allocation'
   last_check=time.monotonic()
  if total>limit:reason='requested_host_memory_exceeded'
  elif time.monotonic()>deadline:reason='test_timeout'
  if reason:break
  time.sleep(.25)
 if not reason:rc=p.wait()
except BaseException:
 reason=reason or 'guard_error';traceback.print_exc()
finally:
 try:
  for proc in parent.children(recursive=True):
   try:owned[proc.pid]=(proc,proc.create_time())
   except psutil.NoSuchProcess:pass
 except psutil.NoSuchProcess:pass
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,signal.SIG_IGN)
 # The dedicated process group may outlive its original leader.
 try:os.killpg(p.pid,signal.SIGTERM)
 except ProcessLookupError:pass
 if p.poll() is None:
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:
   try:os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   p.wait()
  rc=p.returncode
 # Ray may daemonize into another session. Only act on descendants whose
 # process creation time still matches the one observed during this test.
 survivors=[]
 for pid,(proc,born) in owned.items():
  try:
   if pid!=p.pid and proc.is_running() and proc.create_time()==born:proc.terminate();survivors.append(proc)
  except (psutil.NoSuchProcess,psutil.AccessDenied):pass
 _,alive=psutil.wait_procs(survivors,timeout=5)
 for proc in alive:
  try:
   if proc.create_time()==owned[proc.pid][1]:proc.kill()
  except (psutil.NoSuchProcess,psutil.AccessDenied):pass
 report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps({'command':command,'exit_code':rc,'peak_pss_bytes':peak,'memory_limit_bytes':limit,'guard_stop_reason':reason},indent=2))
sys.exit((rc if rc>=0 else 128-rc) if not reason or rc else 1)
