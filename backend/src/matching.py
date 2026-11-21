import os,re,hashlib,json
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import precision_score,recall_score
from sklearn.model_selection import train_test_split
NUM={'零':0,'〇':0,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}

class IncompleteMatch(RuntimeError):
    '''分段比对未能全部完成。调用方必须保留上一份候选名单。'''

def _env_int(name,default):
    try:return max(1,int(os.getenv(name,str(default))))
    except (TypeError,ValueError):return default

def number(text):
 m=re.search(r'([0-9零〇一二三四五六七八九十]+)号',text)
 if not m:return None
 t=m.group(1)
 if t.isdigit():return int(t)
 if '十' in t:
  a,b=t.split('十');return (NUM.get(a,1) if a else 1)*10+(NUM.get(b,0) if b else 0)
 return int(''.join(str(NUM[x]) for x in t))
def normalize(text):
 n=number(text)
 return re.sub(r'([0-9零〇一二三四五六七八九十]+)号',f'{n}号',text) if n is not None else text

# 分段参数：一段别名与整批共享同一套词表/IDF，只是相似度矩阵按段分块计算。
def chunk_size():
 return _env_int('MATCH_CHUNK_SIZE',256)
def segment_workers():
 return _env_int('MATCH_SEGMENT_WORKERS',4)
def cluster_limit():
 return _env_int('MATCH_CLUSTER_MAX_ALIASES',2500)
def segment_count(n,size=None):return (n+ (size or chunk_size()) -1)//(size or chunk_size())

def _bounds(n,size):
 if n==0:return []
 return [(i,min(i+size,n)) for i in range(0,n,size)]

def _starts(ids):
 if len(ids)==0:return np.empty(0,dtype=np.int64)
 return np.concatenate(([0],np.flatnonzero(ids[1:]!=ids[:-1])+1))

def _pair_maxima(block,row_ids,col_ids):
 '''把一块“段×段”相似度压成入口对的最大余弦，行/列都已按入口排序。'''
 rs=_starts(row_ids);cs=_starts(col_ids)
 if len(rs)==0 or len(cs)==0:return {}
 reduced=np.maximum.reduceat(np.maximum.reduceat(block,rs,axis=0),cs,axis=1)
 out={}
 rows=row_ids[rs];cols=col_ids[cs]
 for a,e in enumerate(rows):
  for b,f in enumerate(cols):
   if e==f:continue
   key=(int(e),int(f)) if e<f else (int(f),int(e))
   v=float(reduced[a,b]);prev=out.get(key)
   if prev is None or v>prev:out[key]=v
 return out

def _segment(v,sorted_ids,bounds,start):
 '''一个段：本段别名与自身及其后所有段比对；跨段别名自然在此覆盖。'''
 ps,pe=bounds[start];row_ids=sorted_ids[ps:pe];out={};tiles=0
 for qs,qe in bounds[start:]:
  block=(v[ps:pe]@v[qs:qe].T).toarray()
  for key,val in _pair_maxima(block,row_ids,sorted_ids[qs:qe]).items():
   prev=out.get(key)
   if prev is None or val>prev:out[key]=val
  tiles+=1
 return out,tiles

def _cluster(v,count):
 '''小批量保留原 sklearn 层次聚类标签；大批量跳过全矩阵，避免 O(n^2) 内存。'''
 if count>cluster_limit():return [],False
 sim=(v@v.T).toarray();dist=np.clip(1-sim,0,2);np.fill_diagonal(dist,0)
 labels=AgglomerativeClustering(n_clusters=None,distance_threshold=.55,metric='precomputed',linkage='average').fit_predict(dist)
 return [int(k) for k in labels],True

# Simulated labelled historical alias pairs. Split before TFIDF fit/threshold selection.
def evaluation():
 pairs=[]
 for street in ['长桥路','青石街','河滨路','新民巷','榕树街','学府路','东湖路','西堤路','南苑街','北园街','柳岸路','梅花巷']:
  pairs.extend([(street+'18号',street+'十八号',1),(street+'18号南入口',street+'十八号南门',1),(street+'18号',street+'81号',0),(street+'18号',street+'19号',0)])
 tr,te=train_test_split(list(range(len(pairs))),test_size=.25,stratify=[x[2] for x in pairs],random_state=42)
 vec=TfidfVectorizer(analyzer='char',ngram_range=(1,3));vec.fit([normalize(t) for i in tr for t in pairs[i][:2]])
 def pred(ids,threshold):
  out=[]
  for i in ids:
   a,b,_=pairs[i];v=vec.transform([normalize(a),normalize(b)]);sim=float((v[0]@v[1].T).toarray()[0,0]);out.append(int(number(a)==number(b) and sim>=threshold))
  return out
 threshold=max([.35,.45,.55,.65,.75],key=lambda z:(precision_score([pairs[i][2] for i in tr],pred(tr,z),zero_division=0)+recall_score([pairs[i][2] for i in tr],pred(tr,z),zero_division=0),z))
 y=[pairs[i][2] for i in te];pr=pred(te,threshold)
 return {'seed':42,'train_pairs':len(tr),'holdout_pairs':len(te),'train_ids':tr,'holdout_ids':te,'precision':precision_score(y,pr,zero_division=0),'recall':recall_score(y,pr,zero_division=0),'threshold':threshold,'label_source':'模拟历史别名对；不代表真实地址精度'}

def match(entrances,aliases,progress=None,workers=None,size=None):
 '''整批拟合词权，分段分块比较；跨段别名同样参与，候选与逐对核算一致。'''
 if len(entrances)<2 or len(aliases)<2:raise ValueError('至少两个入口和两个有文字的别名')
 size=size or chunk_size();workers=workers or segment_workers()
 texts=[normalize(a['text']) for a in aliases]
 vec=TfidfVectorizer(analyzer='char',ngram_range=(1,3));v=vec.fit_transform(texts)
 labels,clustered=_cluster(v,len(aliases))
 # 按入口排序让同入口别名相邻，段内可用 reduceat 一次压出入口对最大值。
 order=sorted(range(len(aliases)),key=lambda i:(aliases[i]['entrance_id'],i))
 sv=v[order];sorted_ids=np.array([aliases[i]['entrance_id'] for i in order],dtype=np.int64)
 bounds=_bounds(len(aliases),size);total=len(bounds)
 if total==0:raise IncompleteMatch('没有可比对的别名段')
 expected=total*(total+1)//2
 maxima={};done_tiles=0;done_segments=0
 def merge(out):maxima.update({k:v for k,v in out.items() if k not in maxima or v>maxima[k]})
 def report():
  if progress:progress(done_segments,total)
 try:
  with ThreadPoolExecutor(max_workers=max(1,min(workers,total))) as pool:
   futures={pool.submit(_segment,sv,sorted_ids,bounds,i):i for i in range(total)}
   for f in as_completed(futures):
    i=futures[f]
    try:out,tiles=f.result()
    except Exception as e:raise IncompleteMatch(f'第{i+1}段计算失败：{e}') from e
    if out is None or tiles!=total-i:raise IncompleteMatch(f'第{i+1}段只完成{tiles}/{total-i}块，本批不完整')
    merge(out);done_tiles+=tiles;done_segments+=1;report()
 except IncompleteMatch:
  raise
 except Exception as e:raise IncompleteMatch(f'分段比对失败：{e}') from e
 if done_tiles!=expected or done_segments!=total:raise IncompleteMatch('分段结果不齐，保留原候选名单')
 has_alias={a['entrance_id'] for a in aliases}
 candidates=[]
 for i,x in enumerate(entrances):
  for y in entrances[i+1:]:
   if x['id'] not in has_alias or y['id'] not in has_alias:continue
   key=(x['id'],y['id']) if x['id']<y['id'] else (y['id'],x['id'])
   if key not in maxima:raise IncompleteMatch('入口对缺少分段结果，保留原候选名单')
   score=maxima[key];conflict=x['number']!=y['number'] or x['street']!=y['street']
   if score>=.2:candidates.append({'source_id':x['id'],'target_id':y['id'],'similarity':round(score,6),'conflict':int(conflict)})
 result={'algorithm':'字符TF-IDF+AgglomerativeClustering average linkage','source_sha256':hashlib.sha256(json.dumps([entrances,aliases],ensure_ascii=False,sort_keys=True).encode()).hexdigest(),'evaluation':evaluation(),'chunk_size':size,'segments':total,'clusters':[{'alias_id':a['id'],'label':k} for a,k in zip(aliases,labels)] if clustered else [],'candidates':candidates}
 if not clustered:result['cluster_note']=f'本批{len(aliases)}条别名超出聚类内存上限{cluster_limit()}，簇标签省略；候选门牌对仍按全量分段核算'
 return result

