let report;
const form=document.querySelector('#form'),status=document.querySelector('#status');
async function run(data){
 document.querySelectorAll('button').forEach(b=>b.disabled=true);
 status.textContent='Analyzing normal references and held-out images…';
 document.querySelector('#results').replaceChildren();document.querySelector('#summary').hidden=true;report=null;
 try{
  const response=await fetch('/analyze',{method:'POST',body:data});
  const r=await response.json();if(!response.ok)throw new Error(r.error||'Analysis failed.');
  report=r;document.querySelector('#summary').hidden=false;
  document.querySelector('#metrics').textContent=`${r.synthetic_sample?'SYNTHETIC WASHERS · ':''}${r.reference_count} references · ${r.results.length} held-out images · ${r.elapsed_seconds_excluding_model_load.toFixed(2)} s (model load excluded)`+(r.metrics?` · AUROC ${r.metrics.auroc.toFixed(3)} · Average precision ${r.metrics.average_precision.toFixed(3)}`:' · Ranking metrics require labels containing both classes.');
  for(const row of r.results){const card=document.createElement('article'),title=document.createElement('h3'),score=document.createElement('p');title.textContent=row.name;score.textContent=`Score ${row.score.toFixed(4)}${row.label===undefined?'':` · Label: ${row.label?'defect':'normal'}`}`;card.append(title,score);for(const key of ['original','overlay']){const figure=document.createElement('figure'),img=document.createElement('img'),caption=document.createElement('figcaption');img.src=row[key];img.alt=`${key} of ${row.name}`;caption.textContent=key==='original'?'Input':'Feature-distance overlay';figure.append(img,caption);card.append(figure);}document.querySelector('#results').append(card);}
  status.textContent='Analysis complete. Review scores and images; no classification threshold has been applied.';
 }catch(e){status.textContent=`Unable to analyze: ${e.message}`;}
 finally{document.querySelectorAll('button').forEach(b=>b.disabled=false);}
}
form.addEventListener('submit',e=>{e.preventDefault();run(new FormData(form));});
document.querySelector('#demo').onclick=()=>{const data=new FormData();data.set('sample','true');run(data);};
document.querySelector('#export').onclick=()=>{if(!report)return;const clean={...report,results:report.results.map(({original,overlay,...row})=>row)};const url=URL.createObjectURL(new Blob([JSON.stringify(clean,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='visioninspect-report.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
