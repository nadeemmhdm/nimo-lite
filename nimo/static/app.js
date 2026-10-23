'use strict';

const $ = id => document.getElementById(id);
function setText(id, value) {
  const node = $(id);
  if (!node) {
    console.error(`Nimo Lite dashboard markup is missing #${id}; check that the UI assets are from the same build.`);
    return false;
  }
  node.textContent = String(value ?? '');
  return true;
}
const pretty = value => JSON.stringify(value, null, 2);
const names = {chat:'Chat', training:'Training Center', studio:'Image Labeling Studio', datasets:'Dataset Manager', models:'Model Manager', knowledge:'Local Knowledge Base', video:'Video Preparation', security:'Security & Search', performance:'Performance Monitor'};
const taskNames = {intent:'Basic assistant · NLP intents', text:'Original language model', image:'Image classifier'};
const iconPaths = {
  chat:'M4 4h16v12H8l-4 4V4Z', training:'m12 3 9 5-9 5-9-5 9-5Zm-7 9v5l7 4 7-4v-5',
  studio:'M4 4h16v16H4V4Zm0 12 5-5 5 5 3-3 3 3M8 8h.01', datasets:'M4 6c0-4 16-4 16 0s-16 4-16 0Zm0 0v12c0 4 16 4 16 0V6M4 12c0 4 16 4 16 0',
  models:'m12 3 9 5v9l-9 5-9-5V8l9-5Zm0 10v9M3 8l9 5 9-5', knowledge:'M4 4h6l2 2 2-2h6v15h-6l-2 2-2-2H4V4Zm8 2v15',
  video:'M4 5h12v14H4V5Zm12 5 5-3v10l-5-3', security:'m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6l8-3Zm-4 9 3 3 5-6',
  performance:'M3 18h18M5 14l4-5 4 3 6-8'
};
let token = '', samples = [], state = {}, selected = null, boxes = [], loadedImage = null;
let drawStart = null, suggestion = null, lastChat = null, datasetPage = 0, modelRenderKey = '';
let selectedSamples = new Set(), visibleSamples = [], noticeTimer;
const busy = new Set();

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
function icon(name) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  for (const [key,value] of Object.entries({viewBox:'0 0 24 24', fill:'none', stroke:'currentColor', 'stroke-width':'1.5', 'stroke-linejoin':'round', 'stroke-linecap':'round', 'aria-hidden':'true', class:'nav-icon'})) node.setAttribute(key,value);
  const path = document.createElementNS(node.namespaceURI,'path');
  path.setAttribute('d',iconPaths[name]); node.append(path); return node;
}
function notice(text, error = false) {
  clearTimeout(noticeTimer);
  const node = $('notice');
  if (!node) {
    console.error('Nimo Lite dashboard markup is missing #notice; check that the UI assets are from the same build.');
    return;
  }
  node.textContent = text;
  node.hidden = !text;
  node.classList.toggle('error', error);
  node.setAttribute('role', error ? 'alert' : 'status');
  if (text && !error) noticeTimer = setTimeout(() => { node.hidden = true; }, 12000);
}
async function safe(fn, button) {
  if (button && busy.has(button)) return;
  if (button) {busy.add(button); button.disabled=true; button.setAttribute('aria-busy','true');}
  try {return await fn();}
  catch (error) {notice(error.message || 'Something went wrong. Please try again.',true);}
  finally {if (button) {busy.delete(button); button.disabled=false; button.removeAttribute('aria-busy');}}
}
function bind(id, fn) {$(id).addEventListener('click', event => safe(fn,event.currentTarget));}
function action(label, fn, cls='button-secondary') {
  const node=el('button',label,cls);node.type='button';node.onclick=()=>safe(fn,node);return node;
}
async function api(path, body, method) {
  const options={method:method || (body===undefined?'GET':'POST'),headers:{'X-Nimo-Token':token}};
  if (body!==undefined) {
    if (body instanceof FormData) options.body=body;
    else {options.headers['Content-Type']='application/json'; options.body=JSON.stringify(body);}
  }
  const response=await fetch('/api/'+path,options);
  let data;
  try {data=await response.json();} catch {throw Error('The server returned an unreadable response. Check the local server log.');}
  if (!response.ok) {
    const message=Array.isArray(data.detail) ? data.detail.map(x=>`${x.loc?.slice(1).join('.') || 'Input'}: ${x.msg}`).join('; ') : data.detail;
    throw Error(message || `Request failed (${response.status}).`);
  }
  return data;
}
function empty(title, description, label, fn) {
  const node=el('div',undefined,'empty-state');node.append(el('div','✦','empty-symbol'),el('h3',title),el('p',description));
  if (label) node.append(action(label,fn));return node;
}
function badge(text, status='') {return el('span',text,'badge '+status);}
function stat(label,value,detail) {
  const node=el('div',undefined,'stat');node.append(el('div',label,'stat-label'),el('div',value,'stat-value'),el('div',detail,'stat-detail'));return node;
}
function number(value, digits=3) {return Number.isFinite(value)?value.toFixed(digits):'—';}
function percent(value) {return Number.isFinite(value)?(value*100).toFixed(1)+'%':'—';}
function closeMenu() {$('scrim').hidden=true;document.body.classList.remove('menu-open');$('menuToggle').setAttribute('aria-expanded','false');}
function show(id) {
  notice('');closeMenu();
  for (const key of Object.keys(names)) {
    $(key).hidden=key!==id;
    const button=document.querySelector(`[data-page="${key}"]`);
    button.classList.toggle('active',key===id);
    if (key===id) button.setAttribute('aria-current','page');else button.removeAttribute('aria-current');
    button.querySelector('.nav-active-dot').hidden=key!==id;
  }
  setText('title', names[id]);setText('breadcrumb', names[id]);
  if (['datasets','studio'].includes(id)) safe(refreshSamples);
  if (id==='knowledge') safe(refreshKnowledge);
  if (id==='security') safe(async()=>{setText('audit', pretty(await api('audit')));});
}
for (const [id,title] of Object.entries(names)) {
  const button=action('',()=>show(id),'');button.dataset.page=id;
  button.append(icon(id),el('span',title),el('i',undefined,'nav-active-dot'));$('nav').append(button);
}
for (const button of document.querySelectorAll('[data-go]')) button.onclick=()=>show(button.dataset.go);
for (const button of document.querySelectorAll('[data-prompt]')) button.onclick=()=>{$('prompt').value=button.dataset.prompt;$('prompt').focus();};
$('menuToggle').onclick=()=>{const open=!document.body.classList.contains('menu-open');document.body.classList.toggle('menu-open',open);$('scrim').hidden=!open;$('menuToggle').setAttribute('aria-expanded',String(open));if(open)document.querySelector('#nav button.active').focus();};
$('scrim').onclick=closeMenu;
document.addEventListener('keydown',event=>{if(event.key==='Escape')closeMenu();});
function themeLabel() {const dark=document.documentElement.dataset.theme==='dark';$('themeToggle').setAttribute('aria-label',`Switch to ${dark?'light':'dark'} theme`);$('themeToggle').title=`Switch to ${dark?'light':'dark'} theme`;setText('themeToggle', dark?'☀︎':'◐');}
$('themeToggle').onclick=()=>{const next=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=next;try{localStorage.setItem('nimo-theme',next);}catch{}themeLabel();};
themeLabel();show('chat');
for (const text of ['Text training','Image labeling','Classification','Bounding boxes · export only','Video frames','Q&A','Reasoning & math examples','Coding examples','Review & provenance','Evaluation & history']) $('trainingTypes').append(el('span',text));

function message(who,text,result) {
  const node=el('article',undefined,'message '+(who==='You'?'user':'assistant'));
  const heading=el('div',undefined,'message-heading');heading.append(el('span',who==='You'?'Y':'n','message-avatar'),el('b',who));node.append(heading,el('p',text));
  if(result) {
    const meta=el('div',undefined,'message-meta');
    const label=result.mode.toLowerCase().includes('intent')?'Trained NLP · authored response':result.mode.toLowerCase().includes('knowledge')?'Local knowledge':result.mode.toLowerCase().includes('onboarding')?'Getting started':'Experimental text generation';
    meta.append(el('span',label,'source-tag'),el('span',number(result.latency_s,2)+'s'));
    if(result.confidence!==undefined)meta.append(el('span',percent(result.confidence)+' intent score · uncalibrated'));
    node.append(meta);
    if(result.sources?.length) {
      const sources=el('div',undefined,'message-sources');sources.append(el('b','Sources: '));
      for(const source of [...new Set(result.sources.map(x=>x.source))]) {
        let link;try{const url=new URL(source);if(['https:','http:'].includes(url.protocol)){link=el('a',source);link.href=url.href;link.target='_blank';link.rel='noopener noreferrer';}}catch{}
        sources.append(link || el('span',source),document.createTextNode(' '));
      }
      node.append(sources);
    }
  }
  $('chatWelcome').hidden=true;$('messages').append(node);$('messages').scrollTop=$('messages').scrollHeight;
}
$('chatForm').onsubmit=event=>{
  event.preventDefault();safe(async()=>{
    const text=$('prompt').value.trim();if(!text)return;
    message('You',text);$('prompt').value='';$('prompt').disabled=true;
    try {
      const result=await api('chat',{text,mode:$('chatMode').value});lastChat={question:text,answer:result.answer};
      message('Nimo Lite',result.answer,result);$('feedbackPanel').hidden=false;
    } catch(error){$('prompt').value=text;throw error;}
    finally{$('prompt').disabled=false;$('prompt').focus();}
  },$('sendMessage'));
};
$('prompt').onkeydown=event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();if(!busy.has($('sendMessage')))$('chatForm').requestSubmit();}};
$('chatMode').onchange=()=>{setText('chatModeHelp', {assistant:'Basic intent responses and sourced notes. Experimental model generation is a separate option.',knowledge:'Returns matching saved notes with their sources. No model weights are updated.',model:'Generates from your active original language model. This small experimental model can be incorrect or nonsensical.'}[$('chatMode').value]);};
bind('clearChat',()=>{$('messages').replaceChildren();$('chatWelcome').hidden=false;$('feedbackPanel').hidden=true;lastChat=null;$('prompt').value='';$('prompt').focus();});
bind('feedback',async()=>{if(!lastChat)throw Error('Send a message before rating an answer.');await api('feedback',{...lastChat,rating:$('rating').value,correction:$('correction').value});notice('Feedback saved locally. You can add a reviewed correction in Training Center.');});

$('exampleForm').onsubmit=event=>{event.preventDefault();safe(async()=>{
  const result=await api('examples',{question:$('question').value,answer:$('answer').value,source:$('textSource').value,license:$('textLicense').value,category:$('exampleType').value});
  notice(result.duplicate?'This exact example already exists.':'Example saved for review. No model weights changed.');await refreshSamples();
},$('exampleForm').querySelector('button'));};
bind('loadStarter',async()=>{const result=await api('starter',{});await refreshSamples();$('datasetStatus').value='pending';$('datasetSearch').value='';datasetPage=0;show('datasets');notice(`${result.added} starter examples added for review${result.duplicates?`; ${result.duplicates} duplicates skipped`:''}. Select and approve only the examples you have checked.`);});
bind('focusTraining',()=>{$('trainingRun').scrollIntoView({behavior:'smooth',block:'start'});$('task').focus();});
bind('importDataset',async()=>{const file=$('datasetFile').files[0];if(!file)throw Error('Choose a JSON dataset first.');const form=new FormData();form.append('file',file);const result=await api('import',form);notice(`${result.imported} examples processed. New examples need review before training.`);await refreshSamples();});
function taskHelp() {
  const task=$('task').value;
  setText('taskHelp', {intent:'Learns to recognize basic requests and choose an authored response. It does not generate new answers.',text:'Updates a randomly initialized causal Transformer using answer-focused next-byte training. Small datasets do not create a generally capable assistant.',image:'Trains a separate CNN on approved images with exactly one verified label each. Boxes and OCR are not supported training tasks.'}[task]);
  $('contextSize').disabled=task!=='text';$('epochs').value={intent:60,text:80,image:5}[task];$('validationResult').hidden=true;
}
$('task').onchange=taskHelp;taskHelp();
function renderValidation(result) {
  const panel=$('validationResult');panel.hidden=false;panel.className='validation-panel '+(result.valid?'valid':'invalid');panel.replaceChildren(el('b',result.valid?'✓ Dataset is ready for an experiment':'Some examples need attention'));
  const splits=Object.entries(result.splits).map(([key,value])=>`${key}: ${value.length}`).join(' · ');panel.append(el('p',`${result.samples} approved samples · ${splits}`,'muted'));
  for (const [items,heading] of [[result.errors,'Fix before training'],[result.warnings,'Keep in mind']]) if(items.length){panel.append(el('b',heading));const list=el('ul');for(const item of items)list.append(el('li',item));panel.append(list);}
}
bind('validate',async()=>{renderValidation(await api('validate/'+$('task').value));});
bind('train',async()=>{
  const report=await api('validate/'+$('task').value);renderValidation(report);if(!report.valid)throw Error('Review the dataset issues shown below before training.');
  const result=await api('train',{task:$('task').value,epochs:Number($('epochs').value),resume:$('resumeId').value.trim()||null,context:Number($('contextSize').value)});
  notice('Training started in a separate local worker. You can pause or stop safely between batches.');await poll();
});
for(const button of document.querySelectorAll('[data-command]'))button.onclick=()=>safe(async()=>{
  const job=state.jobs?.find(j=>['queued','running','paused'].includes(j.status));if(!job)throw Error('No active run. To resume an interrupted run, use its checkpoint ID.');
  await api('jobs/'+job.id+'/'+button.dataset.command,{});
  notice({pause:'Pause requested. A checkpoint will be saved at the next batch boundary.',run:'Resume requested.',stop:'Safe stop requested. The latest checkpoint will be retained.',save:'Checkpoint save requested at the next batch boundary.'}[button.dataset.command]);
},button);

async function refreshSamples() {
  samples=await api('samples');const existing=new Set(samples.map(x=>x.id));selectedSamples=new Set([...selectedSamples].filter(x=>existing.has(x)));
  renderSamples();renderImages();renderStats();
}
function selectionState() {
  setText('selectionCount', `${selectedSamples.size} selected`);
  const permitted=selectedSamples.size>0&&$('reviewConfirmed').checked;
  $('approveSelected').disabled=!permitted||busy.has($('approveSelected'));$('rejectSelected').disabled=!permitted||busy.has($('rejectSelected'));
  const checked=visibleSamples.filter(x=>selectedSamples.has(x.id)).length;
  $('selectVisible').checked=visibleSamples.length>0&&checked===visibleSamples.length;$('selectVisible').indeterminate=checked>0&&checked<visibleSamples.length;
}
function renderSamples() {
  const query=$('datasetSearch').value.toLocaleLowerCase().trim(),status=$('datasetStatus').value;
  const filtered=samples.filter(r=>(status==='all'||r.status===status)&&(!query||JSON.stringify(r.content).toLocaleLowerCase().includes(query)||r.source.toLocaleLowerCase().includes(query)));
  const pages=Math.max(1,Math.ceil(filtered.length/12));datasetPage=Math.min(datasetPage,pages-1);visibleSamples=filtered.slice(datasetPage*12,(datasetPage+1)*12);
  $('sampleList').replaceChildren();
  for(const row of visibleSamples) {
    const card=el('article',undefined,'card sample-card'),heading=el('div',undefined,'sample-heading');
    const label=el('label',undefined,'check-label'),checkbox=el('input');checkbox.type='checkbox';checkbox.checked=selectedSamples.has(row.id);checkbox.setAttribute('aria-label','Select '+(row.kind==='text'?row.content.question:row.content.labels.join(', ')||'unlabeled image'));
    checkbox.onchange=()=>{if(checkbox.checked)selectedSamples.add(row.id);else selectedSamples.delete(row.id);selectionState();};label.append(checkbox);
    heading.append(label,el('b',row.kind==='text'?row.content.question:row.content.labels.join(', ')||'Unlabeled image'),badge(row.status,row.status));card.append(heading);
    if(row.kind==='text')card.append(el('p',row.content.answer,'sample-answer'));
    else {const image=el('img',undefined,'sample-thumb');image.src='/api/images/'+row.id;image.alt=row.content.labels.join(', ')||'Unlabeled';image.loading='lazy';card.append(image,el('p',`${row.content.width} × ${row.content.height} · ${row.content.boxes.length} boxes`,'sample-source'));}
    card.append(el('p',`${row.kind==='text'?row.content.category:'Image'} · Source: ${row.source} · Usage: ${row.license} · v${row.version}`,'sample-source'));
    if(row.content.flags?.length)card.append(el('p',row.content.flags.join(' · '),'muted'));
    const actions=el('div',undefined,'sample-actions');
    actions.append(action('Approve',async()=>{let acknowledge_flags=false;if(row.content.flags?.length){acknowledge_flags=confirm('Review these warnings: '+row.content.flags.join('; ')+'. Is this example safe and authorized for training?');if(!acknowledge_flags)return;}await api('samples/'+row.id,{status:'approved',acknowledge_flags},'PATCH');await refreshSamples();notice('Example approved for training.');}),action('Reject',async()=>{await api('samples/'+row.id,{status:'rejected'},'PATCH');await refreshSamples();},'button-ghost'));
    if(row.kind==='image')actions.append(action('Label / correct',()=>{show('studio');selectImage(row);}));
    actions.append(action('Delete',async()=>{if(confirm('Permanently delete this example and its local image files? Existing checkpoints may retain learned information.')){await api('samples/'+row.id,undefined,'DELETE');await refreshSamples();}},'button-ghost'));
    card.append(actions);$('sampleList').append(card);
  }
  if(!filtered.length)$('sampleList').append(empty(samples.length?'No matching examples':'A clean slate to learn from',samples.length?'Try another search or review filter.':'Add your own examples or load the original assistant starter dataset.','Open Training Center',()=>show('training')));
  setText('pageInfo', filtered.length?`Page ${datasetPage+1} of ${pages} · ${filtered.length} examples`:'No examples');$('previousPage').disabled=datasetPage===0;$('nextPage').disabled=datasetPage>=pages-1;selectionState();
}
$('datasetSearch').oninput=$('datasetStatus').onchange=()=>{datasetPage=0;renderSamples();};
$('selectVisible').onchange=()=>{for(const row of visibleSamples){if($('selectVisible').checked)selectedSamples.add(row.id);else selectedSamples.delete(row.id);}renderSamples();};
$('reviewConfirmed').onchange=selectionState;
bind('previousPage',()=>{datasetPage=Math.max(0,datasetPage-1);renderSamples();});bind('nextPage',()=>{datasetPage++;renderSamples();});
for(const [id,status] of [['approveSelected','approved'],['rejectSelected','rejected']])bind(id,async()=>{
  const result=await api('review',{ids:[...selectedSamples],status,reviewed:$('reviewConfirmed').checked});selectedSamples.clear();$('reviewConfirmed').checked=false;await refreshSamples();notice(`${result.changed} examples ${status}.${result.errors.length?' '+result.errors.length+' need individual review; flagged examples were not approved.':''}`);
});
bind('refreshSamples',refreshSamples);
bind('renameCategory',async()=>{if(!$('newCategory').value&&!confirm('Remove this category from annotations and return affected samples to review?'))return;const result=await api('categories',{old:$('oldCategory').value,new:$('newCategory').value});await refreshSamples();notice(`${result.changed} annotations updated and returned to review.`);});

bind('uploadImages',async()=>{
  const files=[...$('imageFiles').files,...$('imageFolder').files];if(!files.length)throw Error('Choose images or an image folder first.');if(!$('imageSource').value.trim()||!$('imageLicense').value.trim())throw Error('Add a source and usage permission before uploading.');
  let duplicates=0,uploaded=0;const errors=[];
  for(const file of files) {
    const form=new FormData();form.append('file',file);form.append('source',$('imageSource').value);form.append('license',$('imageLicense').value);
    try {const result=await api('images',form);duplicates+=Number(result.duplicate);uploaded++;}
    catch(error){errors.push(`${file.name}: ${error.message}`);}
    notice(`Processing images ${uploaded+errors.length}/${files.length}…`);
  }
  await refreshSamples();notice(`${uploaded-duplicates} images added. ${duplicates} duplicates skipped.${errors.length?' '+errors.join(' · '):''}`,errors.length>0);
});
function renderImages() {
  const images=samples.filter(x=>x.kind==='image');setText('imageCount', images.length);$('imageList').replaceChildren();
  for(const row of images) {
    const button=action('',()=>selectImage(row),'');button.classList.toggle('selected',selected?.id===row.id);
    const image=el('img');image.src='/api/images/'+row.id;image.alt=row.content.labels.join(', ')||'Unlabeled image';image.loading='lazy';button.append(image,el('div',row.content.labels.join(', ')||'Needs a label'),badge(row.status,row.status));$('imageList').append(button);
  }
  if(!images.length)$('imageList').append(el('p','Upload an image to start labeling.','muted'));
  const available=Boolean(selected&&images.some(x=>x.id===selected.id));
  for(const id of ['saveLabels','suggest','previousImage','nextImage','clearBoxes','undoBox'])$(id).disabled=!available;
}
function selectImage(row) {
  selected=row;boxes=structuredClone(row.content.boxes);suggestion=null;$('labels').value=row.content.labels.join(', ');$('notes').value=row.content.notes||'';$('annotation').value=row.content.annotation;$('approveImage').checked=row.status==='approved';setText('selectedInfo', `Original ${row.content.width} × ${row.content.height} · ${row.status}`);
  setText('suggestions', 'Request a local classifier suggestion, or label the image yourself.');$('acceptSuggestion').disabled=$('rejectSuggestion').disabled=true;
  const image=new Image();image.onload=()=>{if(selected?.id!==row.id)return;loadedImage=image;const scale=Math.min(1,760/image.width);$('canvas').width=Math.round(image.width*scale);$('canvas').height=Math.round(image.height*scale);draw();};image.src='/api/images/'+row.id;renderImages();
}
function draw(extra) {
  if(!loadedImage)return;const canvas=$('canvas'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(loadedImage,0,0,canvas.width,canvas.height);ctx.strokeStyle='#9d83ff';ctx.fillStyle='#9d83ff';ctx.lineWidth=2;ctx.font='14px sans-serif';
  for(const box of [...boxes,...(extra?[extra]:[])]){ctx.strokeRect(box.x*canvas.width,box.y*canvas.height,box.w*canvas.width,box.h*canvas.height);ctx.fillText(box.label,box.x*canvas.width+3,box.y*canvas.height+16);}setText('boxCount', `${boxes.length} boxes`);
}
function position(event) {const rect=$('canvas').getBoundingClientRect();return {x:Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),y:Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))};}
function box(end) {return {x:Math.min(drawStart.x,end.x),y:Math.min(drawStart.y,end.y),w:Math.abs(end.x-drawStart.x),h:Math.abs(end.y-drawStart.y),label:$('boxLabel').value.trim()};}
$('canvas').onpointerdown=event=>{if(!selected)return;if(!$('boxLabel').value.trim()){notice('Open Bounding boxes and enter a label before drawing.');return;}drawStart=position(event);$('canvas').setPointerCapture(event.pointerId);};
$('canvas').onpointermove=event=>{if(drawStart)draw(box(position(event)));};
$('canvas').onpointerup=event=>{if(!drawStart)return;const item=box(position(event));if(item.w>.002&&item.h>.002)boxes.push(item);drawStart=null;draw();};
$('canvas').onpointercancel=()=>{drawStart=null;draw();};
bind('clearBoxes',()=>{boxes=[];draw();});bind('undoBox',()=>{boxes.pop();draw();});
for(const [id,direction] of [['previousImage',-1],['nextImage',1]])bind(id,()=>{const images=samples.filter(x=>x.kind==='image');if(!images.length)return;const index=images.findIndex(x=>x.id===selected?.id);selectImage(images[(index+direction+images.length)%images.length]);});
bind('saveLabels',async()=>{if(!selected)throw Error('Choose an image first.');await api('samples/'+selected.id,{labels:$('labels').value.split(',').map(x=>x.trim()).filter(Boolean),boxes,notes:$('notes').value,annotation:$('annotation').value,status:$('approveImage').checked?'approved':'pending'},'PATCH');await refreshSamples();selected=samples.find(x=>x.id===selected.id);notice($('approveImage').checked?'Labels saved and image approved for training.':'Labels saved. The image remains pending training approval.');});
bind('suggest',async()=>{if(!selected)throw Error('Choose an image first.');const result=await api('classify/'+selected.id,{});suggestion=result.suggestions[0];setText('suggestions', `Suggested: ${suggestion.label} · ${percent(suggestion.score)} uncalibrated score. This is unverified); compare it with the image before accepting.`;$('acceptSuggestion').disabled=$('rejectSuggestion').disabled=false;});
bind('acceptSuggestion',()=>{if(!suggestion)return;$('labels').value=suggestion.label;$('annotation').value='verified';$('approveImage').checked=false;notice('Suggestion copied into the editor. Check it, then explicitly save and approve.');});
bind('rejectSuggestion',()=>{suggestion=null;setText('suggestions', 'Suggestion rejected. Your existing labels were kept.');$('acceptSuggestion').disabled=$('rejectSuggestion').disabled=true;});

function renderModels() {
  const key=pretty(state.models);if(key===modelRenderKey)return;modelRenderKey=key;$('modelList').replaceChildren();
  for(const model of state.models||[]) {
    const card=el('article',undefined,'card model-card'),heading=el('div',undefined,'card-heading'),title=el('div');title.append(el('span',taskNames[model.task]||model.task,'eyebrow'),el('h2',model.active?'Your active version':'Saved experiment'),el('div',model.id,'model-id'));heading.append(title,badge(model.active?'ACTIVE':model.metrics.accepted?'ACCEPTED':'NOT ACTIVATED',model.active?'active':model.metrics.accepted?'approved':'pending'));card.append(heading);
    const metrics=el('div',undefined,'model-metrics'),test=model.metrics.test,validation=model.metrics.validation;
    for(const [label,value] of [[model.task==='text'?'Test byte accuracy':'Test accuracy',percent(test.accuracy)],['Validation loss',number(validation.loss)],['Test loss',number(test.loss)],[model.task==='text'?'Test bytes':'Test samples',String(test.count)]]){const item=el('div',undefined,'model-metric');item.append(el('span',label),el('b',value));metrics.append(item);}card.append(metrics);
    const row=el('div',undefined,'row');row.append(action('Evaluate held-out test',async()=>{const result=await api('models/'+model.id+'/evaluate',{});notice(`Held-out test: loss ${number(result.metrics.loss)} · ${model.task==='text'?'byte ':''}accuracy ${percent(result.metrics.accuracy)} · ${result.metrics.count} targets.`);}));
    const activate=action(model.active?'Active version':'Restore this version',async()=>{await api('models/'+model.id+'/activate',{});notice('Working version restored. Other experiments were retained.');await poll();});activate.disabled=Boolean(model.active)||!model.metrics.accepted;row.append(activate);
    const download=el('a','Export model ↓','button button-secondary');download.href='/api/models/'+model.id+'/export';download.download='';row.append(download);card.append(row);
    const details=el('details',undefined,'advanced');details.append(el('summary','Full evaluation, split IDs & provenance'),el('pre',pretty(model.metrics)));card.append(details);$('modelList').append(card);
  }
  if(!state.models?.length)$('modelList').append(empty('Your first model starts with examples','Train a basic assistant, text generator, or image classifier. Accepted experiments appear here.','Prepare an experiment',()=>show('training')));
}
async function refreshKnowledge() {
  $('knowledgeList').replaceChildren();const notes=await api('knowledge');
  for(const note of notes){const card=el('article',undefined,'card');card.append(el('p',note.text,'sample-answer'),el('p','Source: '+note.source,'sample-source'),action('Delete note',async()=>{if(confirm('Delete this locally saved note?')){await api('knowledge/'+note.id,undefined,'DELETE');await refreshKnowledge();}},'button-ghost'));$('knowledgeList').append(card);}
  if(!notes.length)$('knowledgeList').append(empty('A memory with a source','Add a note above. Saving information and training a model are different steps.'));
}
$('knowledgeForm').onsubmit=event=>{event.preventDefault();safe(async()=>{await api('knowledge',{text:$('knowledgeText').value,source:$('knowledgeSource').value});notice('Knowledge saved locally. Model weights were not changed.');$('knowledgeText').value='';await refreshKnowledge();},$('knowledgeForm').querySelector('button'));};
$('online').onchange=()=>safe(async()=>{try{const result=await api('network',{online:$('online').checked});notice(result.online?'Wikipedia search enabled. An explicit search sends the query online.':'Offline mode is on.');await poll();}catch(error){$('online').checked=Boolean(state.online);throw error;}});
bind('search',async()=>{const result=await api('search',{text:$('searchQuery').value});$('searchResults').replaceChildren();for(const item of result.results){const paragraph=el('p'),link=el('a',item.title);link.href=item.url;link.target='_blank';link.rel='noopener noreferrer';paragraph.append(link,el('small','Retrieved '+new Date(item.retrieved).toLocaleString()));$('searchResults').append(paragraph);}notice('Sources retrieved. Verify important claims; nothing was added to memory or training.');});
bind('extractVideo',async()=>{const file=$('videoFile').files[0];if(!file)throw Error('Choose a local video first.');const form=new FormData();form.append('file',file);form.append('source',$('videoSource').value);form.append('license',$('videoLicense').value);const result=await api('video',form);setText('videoResult', `${result.frames.length} frames extracted. Open Image Labeling Studio to review them.\n\n`+pretty(result.frames));await refreshSamples();notice('Video frames prepared locally and left pending annotation.');});

function renderStats() {
  const approved=samples.filter(x=>x.status==='approved').length,pending=samples.filter(x=>x.status==='pending').length;
  $('trainingStats').replaceChildren(stat('Your examples',String(samples.length),`${samples.filter(x=>x.kind==='image').length} images · ${samples.filter(x=>x.kind==='text').length} text`),stat('Ready for training',String(approved),`${pending} waiting for review`),stat('Saved model versions',String(state.models?.length||0),`${state.models?.filter(x=>x.active).length||0} active task models`),stat('Available memory',state.hardware?state.hardware.available_gb+' GB':'—','Checked before and during training'));
  if(state.hardware){const h=state.hardware;$('performanceStats').replaceChildren(stat('Available RAM',h.available_gb+' GB',`${h.ram_gb} GB detected limit`),stat('Application memory',h.process_ram_mb+' MB','Current server process · not peak'),stat('CPU usage',h.cpu_percent+'%',`${h.cores} visible logical cores`),stat('Storage available',h.disk_free_gb+' GB','Keep room for checkpoints'));}
}
function renderJob() {
  const job=state.jobs?.[0],active=state.jobs?.find(x=>['queued','running','paused'].includes(x.status));
  for(const button of document.querySelectorAll('[data-command]')) {
    const command=button.dataset.command;
    button.disabled=!active||busy.has(button)||(command==='run'&&active.status!=='paused')||(['pause','save'].includes(command)&&active.status==='paused');
  }
  $('train').disabled=Boolean(active)||busy.has($('train'));
  if(!job)return;
  const p=job.progress,total=p.epochs||job.config.epochs;
  const complete=job.status==='completed',fraction=complete?1:Math.max(0,Math.min(1,((p.epoch||1)-1+(p.batch||0)/(p.batches||1))/total));
  setText('jobBadge', job.status.replaceAll('_',' '));$('jobBadge').className='badge '+job.status;
  setText('jobTitle', `${taskNames[job.task]} · ${job.status}`);$('progress').value=fraction*100;setText('progressLabel', Math.round(fraction*100)+'%');
  setText('jobSummary', p.error || (complete?(p.metrics?.accepted?'Validation criteria passed. This version is now available in Model Manager.':'Experiment saved. Acceptance criteria were not met, so your active version was retained.'):`Epoch ${p.epoch||0} of ${total} · Batch ${p.batch||0} of ${p.batches||'…'}${p.device?' · '+p.device.toUpperCase():''}. Commands apply between batches.`));
  $('jobMetrics').replaceChildren();for(const [label,value] of [['Training loss',number(p.loss)],['Validation loss',number(p.validation?.loss)],['Duration',p.duration_s?number(p.duration_s,1)+'s':'—']]){const node=el('div',label,'mini-metric');node.append(el('b',value));$('jobMetrics').append(node);}setText('job', pretty(job));
}
async function poll() {
  state=await api('status');const h=state.hardware;setText('resources', `${h.available_gb} GB free · ${state.online?'ONLINE':'OFFLINE'}`);$('online').checked=state.online;setText('hardware', pretty(h));renderStats();renderJob();renderModels();
}
(async()=>{
  try {const result=await api('bootstrap');token=result.token;await Promise.all([poll(),refreshSamples(),refreshKnowledge()]);const starter=await api('starter');setText('starterCount', starter.examples);setInterval(()=>{if(!document.hidden)safe(poll);},2500);}
  catch(error){notice(error.message,true);}
})();
