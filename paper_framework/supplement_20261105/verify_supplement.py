"""Read-only verification of the transferred supplementary experiment archives.

Reproduce cell coverage from raw results, independently re-score selected PNGs,
retain all failure/exclusion records, and compare the transferred analysis.
This script never executes generated model scripts or makes API calls.
"""
from pathlib import Path, PureWindowsPath
from collections import Counter
import argparse, hashlib, importlib.util, itertools, json
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
TASKS=['global','csc','dzh','seto','arctic01','hecate']
MM={'accuracy_pct':'accuracy_pct','hss':'hss','boundary_f1':'boundary_f1_d1','island_iou_median':'island_iou_median','island_iou_area_weighted':'island_iou_area_weighted','narrow_water_keep':'narrow_water_keep'}
SCALE={m:(1 if m=='accuracy_pct' else 100) for m in MM}

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def relocate(p,archive,run_dir):
    if not p:return None
    parts=PureWindowsPath(p).parts
    if archive.name in parts:
        return archive.joinpath(*parts[parts.index(archive.name)+1:])
    if 'outputs' in parts:return run_dir.joinpath(*parts[parts.index('outputs'):])
    return run_dir/PureWindowsPath(p).name

def extract(source):
    rows=[]; disagreements=[]; snapshots=[]
    for ar in sorted(source.glob('lowhigh_*')):
        if not ar.is_dir():continue
        chunks=ar.name.split('_'); batch=chunks[1]; provider=chunks[2]; arm=chunks[3]
        mpath=ar/'manifest.json'; doc=read(mpath)
        snapshots.append({'path':str(mpath),'sha256':sha(mpath),'run_count':len(doc['runs'])})
        lookup={r['run_id']:r for r in doc['runs']}
        for rd in sorted((ar/'_runs').iterdir()):
            if not rd.is_dir():continue
            rp=rd/'result.json'; prov=read(rd/'provenance.json') if (rd/'provenance.json').exists() else {}
            if rp.exists():
                r=read(rp)
                if r['run_id'] in lookup:
                    old=lookup[r['run_id']]
                    for key in ['status','map_metrics','reasoning_effort','model','task','group']:
                        if r.get(key)!=old.get(key):disagreements.append({'run_id':r['run_id'],'field':key})
            else:
                # The interrupted runs are identified in supplied attempt table;
                # their existence and lack of a result are independent evidence.
                r={'run_id':rd.name,'status':'interrupted_no_result','task':None,'group':None}
                match=next((t for t in TASKS if '_'+t+'_g' in rd.name),None)
                r['task']=match
                import re
                g=re.search(r'_g(\d+)_',rd.name)
                r['group']=int(g.group(1)) if g else None
            mm=r.get('map_metrics') or {}
            row={'batch':batch,'provider':provider,'arm':arm,'archive':ar.name,'run_id':r['run_id'],
                 'status':r.get('status'),'task':r.get('task'),'group':r.get('group'),
                 'started_at':r.get('started_at',''),'finished_at':r.get('finished_at',''),
                 'protocol_valid':prov.get('protocol_valid',True),'exclusion_reason':prov.get('exclusion_reason',''),
                 'backend':r.get('backend'),'effort':r.get('reasoning_effort'),
                 'transport':r.get('transport'),'duration_sec':r.get('duration_sec'),
                 'validation_status':(r.get('validation') or {}).get('status'),
                 'error':r.get('error',''),'source_run_dir':str(rd),
                 'expected_effort':'low' if arm=='L' else 'high',
                 'model':r.get('model'),'prompt_sha256':r.get('prompt_sha256')}
            for k,key in MM.items():row[k]=mm.get(key,np.nan)
            rows.append(row)
    return pd.DataFrame(rows),disagreements,snapshots

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);args=ap.parse_args();source=Path(args.source)
    df,disagreements,snapshots=extract(source)
    df.to_csv(OUT/'all_attempts_verified.csv',index=False,encoding='utf-8-sig')
    valid=df[(df.status=='ok')&df.protocol_valid&(df.validation_status=='ok')&(df.backend=='api')&(df.effort==df.expected_effort)&df[list(MM)].notna().all(axis=1)].copy()
    keys=['batch','provider','arm','task','group']
    selected=valid.sort_values(['started_at','run_id']).drop_duplicates(keys,keep='first').copy()
    selected.to_csv(OUT/'selected_cells_verified.csv',index=False,encoding='utf-8-sig')
    groupkeys=['batch','provider','arm','task']
    clean=selected.groupby(groupkeys)[list(MM)].mean().reset_index()
    clean['n']=selected.groupby(groupkeys).size().to_numpy()
    clean.to_csv(OUT/'clean_cell_means.csv',index=False,encoding='utf-8-sig')
    clean.groupby(['batch','provider','arm'])[list(MM)].mean().reset_index().to_csv(OUT/'clean_equal_map_arm_means.csv',index=False,encoding='utf-8-sig')
    coverage=[];missing=[]
    for ar,g in df.groupby('archive'):
        batch=g.batch.iloc[0]; arm=g.arm.iloc[0]
        ts=TASKS if batch=='6map' else ['seto','hecate'];gs=[3,4] if arm=='Hsham' else [1,2]
        got=selected[selected.archive==ar]
        for t,group in itertools.product(ts,gs):
            if not ((got.task==t)&(got.group==group)).any():missing.append({'archive':ar,'task':t,'group':group})
        coverage.append({'archive':ar,'target_cells':len(ts)*len(gs),'attempts':len(g),
                         'ok':int((g.status=='ok').sum()),'partial':int((g.status=='partial').sum()),
                         'failed':int((g.status=='failed').sum()),'interrupted':int((g.status=='interrupted_no_result').sum()),
                         'excluded':int((~g.protocol_valid).sum()),'complete_cells':len(got),
                         'raw_ok_rate':float((g.status=='ok').mean())})
    coverage=pd.DataFrame(coverage);coverage.to_csv(OUT/'coverage_verified.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(missing,columns=['archive','task','group']).to_csv(OUT/'missing_cells_verified.csv',index=False,encoding='utf-8-sig')
    supplied=pd.read_csv(source/'runscripts_analysis/selected_cells.csv')
    set_match=set(supplied.run_id)==set(selected.run_id)

    # Validate actual files and re-score each selected run using current frozen scorer.
    spec=importlib.util.spec_from_file_location('frozen_quant',ROOT/'quant/quantify.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    task_config=read(ROOT/'experiment_config/tasks.json'); refs={t:np.loadtxt(ROOT/c['ref'],delimiter=',',dtype=np.uint8) for t,c in task_config.items()}
    checks=[]; metric_errors=[]
    for _,s in selected.iterrows():
        ar=source/s.archive;rd=Path(s.source_run_dir);r=read(rd/'result.json'); cand=r.get('candidate') or {}; af=(r.get('archive') or {}).get('files') or {}
        def asset(key):
            raw=cand.get({'png':'source','csv':'csv','upscale':'upscale'}[key])
            pp=relocate(raw,ar,rd)
            if pp and pp.exists():return pp
            pp=relocate(af.get(key),ar,rd)
            return pp
        pp,cp,up=asset('png'),asset('csv'),asset('upscale')
        if pp is None or not pp.exists():
            checks.append({'run_id':s.run_id,'png_exists':False});continue
        with Image.open(pp) as im:P=np.array(im.convert('L')); shape_ok=P.shape==refs[s.task].shape;binary=bool(np.isin(P,[0,255]).all());P=(P>127.5).astype(np.uint8)
        C=np.loadtxt(cp,delimiter=',',dtype=np.uint8) if cp and cp.exists() else None
        U=np.array(Image.open(up).convert('L')) if up and up.exists() else None
        target=np.repeat(np.repeat(P*255,8,axis=0),8,axis=1)
        mm=r['map_metrics'];R=refs[s.task];c=q.component_stats(P,R,2,0.3);nw,nw_n=q.narrow_water_preservation(P,R,2)
        recal={'accuracy_pct':float(q.accuracy(P,R)*100),'hss':float(q.hss(P,R)),
               'boundary_f1_d1':float(q.boundary_f1(P,R,1)[2]),'narrow_water_keep':nw,
               'narrow_water_cells':nw_n,**c}
        deltas={k:abs(float(v)-float(mm[k])) for k,v in recal.items() if k in mm}
        maxdelta=max(deltas.values(),default=0)
        if maxdelta>1e-6:metric_errors.append({'run_id':s.run_id,'deltas':{k:v for k,v in deltas.items() if v>1e-6}})
        checks.append({'run_id':s.run_id,'png_exists':True,'shape_ok':shape_ok,'binary_ok':binary,
                       'csv_agrees':C is not None and np.array_equal(P,C),'upscale_agrees':U is not None and np.array_equal(U,target),
                       'png_hash_matches':sha(pp)==cand.get('sha256'),'metric_max_abs_error':maxdelta})
    check_df=pd.DataFrame(checks);check_df.to_csv(OUT/'artifact_rescore_checks.csv',index=False,encoding='utf-8-sig')

    # H/L pair by replicate; Hsham 3/4 is relabeled 1/2 only within sham comparison.
    pairs=[]
    idx=selected.set_index(keys)
    for (batch,provider),g in selected.groupby(['batch','provider']):
        for _,h in g[g.arm=='H'].iterrows():
            for other,name,offset in [('L','L_minus_H',0),('Hsham','sham_minus_H',2)]:
                k=(batch,provider,other,h.task,h.group+offset)
                if k not in idx.index:continue
                v=idx.loc[k];row={'batch':batch,'provider':provider,'contrast':name,'task':h.task,'replicate':h.group,
                                  'high_run':h.run_id,'other_run':v.run_id,
                                  'high_started':h.started_at,'other_started':v.started_at}
                for m in MM:row[m]=float(v[m]-h[m])
                pairs.append(row)
    pairs=pd.DataFrame(pairs);pairs.to_csv(OUT/'pairs_verified.csv',index=False,encoding='utf-8-sig')
    permap=pairs.groupby(['batch','provider','contrast','task'])[list(MM)].mean().reset_index()
    permap.to_csv(OUT/'per_map_effects_verified.csv',index=False,encoding='utf-8-sig')
    rng=np.random.default_rng(20261103);effects=[]
    for (batch,provider),g in pairs.groupby(['batch','provider']):
        for m in MM:
            hl=g[g.contrast=='L_minus_H'];sh=g[g.contrast=='sham_minus_H'];x=hl.groupby('task')[m].mean().to_numpy();sx=sh.groupby('task')[m].mean().to_numpy()
            B=x[rng.integers(0,len(x),size=(20000,len(x)))].mean(axis=1)
            effects.append({'batch':batch,'provider':provider,'metric':m,'hl_pairs':len(hl),'maps':len(x),
                            'mean_map_equal_L_minus_H':float(x.mean()),'ci95_low':float(np.quantile(B,.025)),
                            'ci95_high':float(np.quantile(B,.975)),'negative_maps':int((x<0).sum()),
                            'sham_pairs':len(sh),'sham_abs_pair_p95':float(sh[m].abs().quantile(.95)) if len(sh) else None,
                            'sham_map_abs_p95':float(np.quantile(abs(sx),.95)) if len(sx) else None,
                            'sham_map_signed_mean':float(sx.mean()) if len(sx) else None})
    eff=pd.DataFrame(effects);eff.to_csv(OUT/'effects_verified.csv',index=False,encoding='utf-8-sig')
    previous=pd.read_csv(source/'runscripts_analysis/effects.csv');diff=eff.merge(previous,on=['batch','provider','metric'],suffixes=('_verified','_supplied'))
    same_effect=bool(np.allclose(diff.mean_map_equal_L_minus_H_verified,diff.mean_map_equal_L_minus_H_supplied,atol=1e-9))
    # Response files: deduplicate by path at run level, never count archive copies.
    response_counts=Counter();cost_n=0;tokens=[0,0]
    for rd in df.source_run_dir:
        f=Path(rd)/'response.json'
        if f.exists():
            obj=read(f);u=obj.get('usage') or {};response_counts['files']+=1
            cost_n+=u.get('cost') is not None
            tokens[0]+=int(u.get('input_tokens') or 0);tokens[1]+=int(u.get('output_tokens') or 0)
    boolcols=['png_exists','shape_ok','binary_ok','csv_agrees','upscale_agrees','png_hash_matches']
    result={'source':str(source),'attempts':len(df),'status_counts':dict(Counter(df.status)),
            'complete_cells':len(selected),'missing_cells':len(missing),'6map_complete_cells':int((selected.batch=='6map').sum()),
            'backfill_complete_cells':int((selected.batch=='backfill').sum()),'excluded_attempts':int((~df.protocol_valid).sum()),
            'valid_ok_runs_including_duplicates':len(valid),'selected_ids_match_supplied':set_match,
            'manifest_result_disagreements':disagreements,'rescore_runs':len(check_df),'rescore_metric_errors':metric_errors,
            'artifact_all_pass':bool(check_df[boolcols].all().all()),'mean_effects_match_supplied':same_effect,
            'pair_counts':pairs.groupby(['batch','provider','contrast']).size().reset_index(name='n').to_dict('records'),
            'final_response_files':response_counts['files'],'reported_cost_count':cost_n,'final_input_tokens':tokens[0],
            'final_output_tokens':tokens[1],'cost_conclusion':'unknown; no usage.cost returned',
            'input_manifests':snapshots}
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 补充实验独立复核（2026-11-05）','',f'源目录：`{source}`','',
           f'- 原始尝试 {len(df)}：{dict(Counter(df.status))}。',
           f'- 六图主批次完整格子 {int((selected.batch=="6map").sum())}/72；两图独立补充 {int((selected.batch=="backfill").sum())}/16；缺格 {len(missing)}。',
           f'- 协议排除 {int((~df.protocol_valid).sum())} 次；同格子取最早有效完整成功。与随包分析选中 ID 一致：{set_match}。',
           f'- 对 {len(check_df)} 个选中输出重新检查 PNG/CSV/8倍图与哈希，全部通过：{result["artifact_all_pass"]}；重新评分不一致 {len(metric_errors)}。',
           f'- 效应均值与随包报告一致：{same_effect}。','',
           '## 六图主批次（L−H，六图等权；统一以百分点表示）','',
           '| 渠道 | 指标 | 配对数 | 平均差/百分点 | 下降地图 | sham 配对绝对差 P95/百分点 |','|---|---|---:|---:|---:|---:|']
    labels={'accuracy_pct':'准确率','hss':'HSS','boundary_f1':'边界 F1@1','island_iou_median':'分量 IoU 中位数','island_iou_area_weighted':'分量格点加权 IoU','narrow_water_keep':'近陆窄水保留率'}
    for _,r in eff[eff.batch=='6map'].iterrows():
        sc=SCALE[r.metric]
        lines.append(f'| {r.provider} | {labels[r.metric]} | {int(r.hl_pairs)} | {r.mean_map_equal_L_minus_H*sc:+.4f} | {int(r.negative_maps)}/6 | {r.sham_abs_pair_p95*sc:.4f} |')
    lines+=['','## 判读边界','',
            '- 覆盖补齐与方向性假设获支持是不同问题。逐渠道、逐指标报告，不能合并两渠道宣称普遍降低。',
            '- Hsham 的 3/4 对应 H 的 1/2 仅为重复编号重标记，分别保留原始组号。backfill 不混入六图批次。',
            '- 随包报告用六图平均 L−H 对单次 sham 绝对差 P95；二者聚合尺度不同。这只是描述性幅度参照，不能据此宣布通过/未通过检测器性能验证。',
            '- high/low 对比只有每图 2 重复，sham 单次配对每渠道 12 个；时序、补跑和失败选择限制因果解释。',
            '- 所有失败与中断应保留在交付率和成本口径中。没有 usage.cost 时实际账单未知，不能再次宣称已验证每模型 10 美元以内。',
            '- 原稿关于 2026-10-29 旧批次未完成的记载仍是历史事实；本轮 2026-11-03 至 05 数据应作为独立新增批次。',
            '- C H 自动 summary/Excel 仍计入14个原始ok（全球/南海各3次），含已排除的错路由成功与重复成功；主分析应采用选中12格的清洗表。','',
            '## 输出','',
            '- verification.json：独立校验结果与输入清单哈希。',
            '- all_attempts_verified.csv / selected_cells_verified.csv：全部尝试和选中完整格子。',
            '- coverage_verified.csv / missing_cells_verified.csv：覆盖与缺格。',
            '- artifact_rescore_checks.csv：逐输出文件与指标复核。',
            '- pairs_verified.csv / per_map_effects_verified.csv / effects_verified.csv：原始配对与效应。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='input_manifests'},ensure_ascii=False,indent=2));print('\nCOVERAGE\n',coverage.to_string(index=False));print('\nPRIMARY RESULTS\n',eff[(eff.batch=='6map')&eff.metric.isin(['accuracy_pct','boundary_f1','island_iou_median','island_iou_area_weighted'])].to_string(index=False))

if __name__=='__main__':main()