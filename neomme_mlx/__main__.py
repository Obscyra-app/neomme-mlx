"""Local indexing and retrieval CLI. No network or model downloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
from PIL import Image


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def documents(paths, max_side, max_pages):
    for raw in paths:
        path=Path(raw)
        suffix=path.suffix.lower()
        if suffix=='.pdf':
            try:import pypdfium2 as pdfium
            except ImportError as e:raise RuntimeError('PDF support requires pypdfium2') from e
            pdf=pdfium.PdfDocument(str(path))
            try:
                if len(pdf)>max_pages:
                    raise ValueError(f'{path.name}: {len(pdf)} pages exceeds --max-pages={max_pages}')
                for i in range(len(pdf)):
                    page=pdf[i]
                    try:
                        w,h=page.get_size()
                        bitmap=page.render(scale=max_side/max(w,h))
                        try:image=bitmap.to_pil().convert('RGB').copy()
                        finally:bitmap.close()
                    finally:page.close()
                    yield {'source':str(path),'page':i+1,'kind':'pdf'}, {'image':image}
            finally:pdf.close()
        elif suffix in ('.txt','.md'):
            content=path.read_text(encoding='utf-8')
            # Paragraph-preserving chunks with a bounded fallback for long paragraphs.
            parts=[];current=''
            for paragraph in content.split('\n\n'):
                for start in range(0,max(1,len(paragraph)),1200):
                    fragment=paragraph[start:start+1200]
                    if current and len(current)+len(fragment)>1600:
                        parts.append(current);current=''
                    current+=('\n\n' if current else '')+fragment
            if current or not parts:parts.append(current)
            for i,text in enumerate(parts):
                yield {'source':str(path),'chunk':i+1,'kind':'text','preview':text[:240]},text
        elif suffix in ('.png','.jpg','.jpeg','.webp','.bmp','.tif','.tiff'):
            yield {'source':str(path),'kind':'image'},{'image':str(path)}
        else:raise ValueError(f'Unsupported file extension: {suffix}')


def main():
    p=argparse.ArgumentParser(description='NeoMME document retrieval on Apple Silicon')
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--dtype',choices=['float32'],default='float32',help='Validated precision; FP16 did not pass all parity gates')
    sub=p.add_subparsers(dest='command',required=True)
    index=sub.add_parser('index')
    index.add_argument('files',nargs='+')
    index.add_argument('--output',required=True,type=Path)
    index.add_argument('--max-side',type=int,default=1024)
    index.add_argument('--max-pages',type=int,default=100)
    search=sub.add_parser('search')
    search.add_argument('--index',required=True,type=Path)
    search.add_argument('--query',required=True)
    search.add_argument('--mode',choices=['late','dense'],default='late')
    search.add_argument('--top-k',type=int,default=5)
    a=p.parse_args()
    from .model import NeoMME
    from .processor import Processor
    from .retrieval import rank
    import mlx.core as mx
    mx.set_cache_limit(64*1024**2)
    fingerprint=sha(a.model/'model.safetensors')
    assets={name:sha(a.model/name) for name in ['config.json','tokenizer.json','processor_config.json','tokenizer_config.json']}
    if a.command=='search':
        meta=json.loads((a.index/'index.json').read_text())
        if meta['weights_sha256']!=fingerprint or meta['dtype']!=a.dtype or meta.get('assets_sha256')!=assets:
            raise ValueError('Index belongs to different weights, processor assets or precision; rebuild it')
        if a.top_k<1:raise ValueError('top-k must be positive')
    else:
        if a.output.exists():raise FileExistsError('Index output already exists; choose a new directory')
        if a.max_pages<1:raise ValueError('max-pages must be positive')
    processor=Processor(a.model,max_side=a.max_side if a.command=='index' else 1024)
    start=time.perf_counter();model=NeoMME(a.model,a.dtype);load_seconds=time.perf_counter()-start
    if a.command=='index':
        a.output.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='.neomme-index-',dir=a.output.parent))
        records=[];start=time.perf_counter()
        try:
            for record,item in documents(a.files,a.max_side,a.max_pages):
                inputs=processor([item],task='document');out=model(**inputs)
                filename=f'{len(records):06d}.npz'
                np.savez(staging/filename,dense=np.asarray(out['dense'][0]),multi=np.asarray(out['multi'][0]))
                record.update(file=filename,sha256=sha(staging/filename),tokens=int(inputs['attention_mask'].sum()))
                records.append(record)
            if not records:raise ValueError('No documents')
            metadata={'format_version':1,'weights_sha256':fingerprint,'assets_sha256':assets,'dtype':a.dtype,'max_side':a.max_side,'scoring':'MeanMaxSim / dense cosine','documents':records}
            (staging/'index.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
            staging.rename(a.output)
        except BaseException:
            shutil.rmtree(staging);raise
        print(json.dumps({'index':str(a.output),'documents':len(records),'load_seconds':load_seconds,'index_seconds':time.perf_counter()-start},ensure_ascii=False))
    else:
        start=time.perf_counter();inputs=processor([a.query],task='query');out=model(**inputs)
        query={k:np.asarray(v[0]) for k,v in out.items()};docs=[]
        for record in meta['documents']:
            filename=record['file']
            if Path(filename).name!=filename:raise ValueError('Invalid index entry path')
            path=a.index/filename
            if sha(path)!=record['sha256']:raise ValueError('Index checksum mismatch')
            with np.load(path,allow_pickle=False) as z:docs.append({k:z[k] for k in ['dense','multi']})
        ranked=rank(query,docs,a.mode)[:a.top_k]
        hits=[{**meta['documents'][i],'score':score} for i,score in ranked]
        print(json.dumps({'query':a.query,'mode':a.mode,'hits':hits,'load_seconds':load_seconds,'search_seconds':time.perf_counter()-start},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
