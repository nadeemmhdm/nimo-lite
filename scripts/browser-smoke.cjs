// Optional developer UI check: requires Playwright and Chromium installed locally.
const {chromium}=require('playwright');
const fs=require('fs');
(async()=>{
 const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8080');
 await page.waitForFunction(()=>document.querySelector('#resources').textContent.includes('OFFLINE'));
 await page.evaluate(async()=>{const {token}=await(await fetch('/api/bootstrap')).json();const notes=await(await fetch('/api/knowledge')).json();for(const n of notes)if(n.source==='Original browser test fixture')await fetch('/api/knowledge/'+n.id,{method:'DELETE',headers:{'X-Nimo-Token':token}});});
 await page.locator('[data-page="knowledge"]').click();
 await page.locator('#knowledgeText').fill('Nimo Lite keeps this browser smoke note locally.');
 await page.locator('#knowledgeSource').fill('Original browser test fixture');
 await page.locator('#knowledgeForm button').click();
 await page.waitForFunction(()=>document.querySelector('#knowledgeList').textContent.includes('browser smoke note'));
 await page.locator('[data-page="chat"]').click();
 await page.locator('#prompt').fill('browser smoke note');await page.locator('#chatForm button').click();
 await page.waitForFunction(()=>document.querySelector('#messages').textContent.includes('Knowledge')||document.querySelector('#messages').textContent.includes('weights unchanged'));
 await page.locator('[data-page="studio"]').click();
 await page.locator('#imageSource').fill('Original synthetic browser test');await page.locator('#imageLicense').fill('CC0');
 await page.locator('#imageFiles').setInputFiles('/tmp/nimo-browser.png');await page.locator('#uploadImages').click();
 await page.locator('#imageList button').first().waitFor();await page.locator('#imageList button').first().click();
 await page.waitForFunction(()=>document.querySelector('#canvas').width===40);
 await page.locator('#labels').fill('red');await page.locator('#notes').fill('Synthetic image, not real-world benchmark');await page.locator('#annotation').selectOption('verified');await page.locator('#approveImage').check();
 await page.locator('#saveLabels').click();await page.waitForFunction(()=>document.querySelector('#notice').textContent.includes('approved'));
 await page.locator('[data-page="datasets"]').click();await page.waitForFunction(()=>document.querySelector('#sampleList').textContent.includes('red'));
 const downloadPromise=page.waitForEvent('download');await page.locator('a[href="/api/export"]').click();const download=await downloadPromise;
 if(download.suggestedFilename()!=='nimo-dataset.zip')throw Error('Wrong export');
 for(const id of ['training','models','video','security','performance']){await page.locator(`[data-page="${id}"]`).click();if(!await page.locator('#'+id).isVisible())throw Error('Missing page '+id);}
 await page.locator('[data-page="chat"]').click();
 fs.mkdirSync('docs/screenshots',{recursive:true});await page.screenshot({path:'docs/screenshots/workbench.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});await page.locator('[data-page="studio"]').click();if(!await page.locator('#studio').isVisible())throw Error('Mobile navigation failed');
 if(errors.length)throw Error(errors.join('\n'));
 console.log('PASS: desktop/mobile navigation, local knowledge chat, image upload/label approval, export, no JavaScript errors');
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
