const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE||undefined,args:['--disable-dev-shm-usage','--disable-gpu']});
 try{
 const page=await browser.newPage({viewport:{width:1280,height:900}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:8769');
 await page.getByRole('button',{name:'Try synthetic washer sample'}).click();
 await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Analysis complete'),{},{timeout:120000});
 assert.equal(await page.locator('article').count(),24);
 assert.match(await page.locator('#metrics').innerText(),/SYNTHETIC WASHERS/);
 const downloaded=page.waitForEvent('download');await page.getByRole('button',{name:'Export report JSON'}).click();
 const download=await downloaded;const report=JSON.parse(fs.readFileSync(await download.path(),'utf8'));
 assert.equal(report.results.length,24);assert.equal(report.results[0].original,undefined);assert.equal(report.synthetic_sample,true);
 await page.screenshot({path:'outputs/browser-desktop.png'});
 // Exercise actual multipart upload, not only the demo endpoint.
 await page.locator('input[name=reference]').setInputFiles(['outputs/sample/reference/reference_00.png','outputs/sample/reference/reference_01.png']);
 await page.locator('input[name=evaluation]').setInputFiles(['outputs/sample/evaluation/normal_00.png','outputs/sample/evaluation/defect_12.png']);
 await page.locator('textarea').fill('{"normal_00.png":0,"defect_12.png":1}');
 await page.getByRole('button',{name:'Analyze images',exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Analysis complete'));
 assert.equal(await page.locator('article').count(),2);
 await page.locator('textarea').fill('{');await page.getByRole('button',{name:'Analyze images',exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Unable to analyze'));
 assert.equal(await page.locator('article').count(),0);
 await page.setViewportSize({width:390,height:844});
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 await page.screenshot({path:'outputs/browser-mobile.png'});
 assert.deepEqual(errors,[]);console.log('Browser passed: real-model sample, multipart upload, JSON export, malformed labels, mobile layout; zero page errors.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
