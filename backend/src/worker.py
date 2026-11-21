import os,time,pathlib,logging
import redis
from sqlalchemy import update
from .database import Session
from .models import MatchRun,Entrance,Alias,Candidate
from .common import rows,dump
from .matching import match,IncompleteMatch

def progress(run_id,done,total):
 with Session() as s:
  s.execute(update(MatchRun).where(MatchRun.id==run_id,MatchRun.state=='running').values(progress=done,segments=total))
  s.commit()

def claim(run_id):
 with Session() as s:
  ok=s.execute(update(MatchRun).where(MatchRun.id==run_id,MatchRun.state=='queued').values(state='running',progress=0,segments=0,error='')).rowcount
  s.commit();return ok==1

def finish(run_id,result):
 '''只有全部段都算齐才写入候选并置 ready；上一份完整名单在此之前不受影响。'''
 with Session() as s:
  run=s.get(MatchRun,run_id)
  if not run or run.state!='running':return
  run.result=result;run.progress=run.segments;run.state='ready'
  s.add_all([Candidate(run_id=run.id,**d) for d in result['candidates']])
  s.commit()

def fail(run_id,error):
 with Session() as s:
  run=s.get(MatchRun,run_id)
  if not run:return
  run.state='failed';run.error=str(error)[:500];run.progress=min(run.progress or 0,run.segments or 0)
  s.commit()

def execute(run_id):
 if not claim(run_id):return
 try:
  with Session() as s:
   entrances=[dump(x) for x in rows(s,Entrance)];aliases=[dump(x) for x in rows(s,Alias)]
  result=match(entrances,aliases,progress=lambda done,total:progress(run_id,done,total))
  finish(run_id,result)
 except IncompleteMatch as e:
  logging.warning('matching incomplete: %s',e);fail(run_id,f'分段结果不齐，沿用上次完整名单：{e}')
 except Exception as e:
  logging.exception('matching failed');fail(run_id,str(e)[:500])

def main():
 q=redis.from_url(os.environ['REDIS_URL'],decode_responses=True)
 while True:
  pathlib.Path('/artifacts/address-heartbeat').write_text(str(time.time()))
  item=q.blpop('address:matching',timeout=2)
  if not item:continue
  try:execute(int(item[1]))
  except (TypeError,ValueError):logging.warning('ignored malformed queue item: %r',item)

if __name__=='__main__':main()
