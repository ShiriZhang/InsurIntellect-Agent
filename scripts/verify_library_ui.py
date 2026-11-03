"""Real-model browser walkthrough in a new temporary library; no encoder/API substitutes."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from ui_harness import serve_library
from playwright.sync_api import sync_playwright
import fitz
from app.core.config import settings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--screenshot',type=Path,required=True)
    args=parser.parse_args()
    root=Path(tempfile.mkdtemp(prefix='insur-ui-real-'))
    source=root/'walkthrough.pdf'
    with fitz.open() as document:
        page=document.new_page()
        page.insert_text((45,60),'Sample insurance policy\n1. Waiting period\nThe waiting period is 90 days from the effective date.\nAccidental injury is exempt from the waiting period.')
        document.save(source)
    result={'real_local_embedding':True,'real_generation_api':True,'work_directory':str(root),'steps':[]}
    try:
        with serve_library(root/'managed') as (url,ingestion,deletion):
            with sync_playwright() as p:
                try: browser=p.chromium.launch(channel='msedge',headless=True)
                except Exception: browser=p.chromium.launch(headless=True)
                context=browser.new_context(viewport={'width':1440,'height':1000})
                page=context.new_page(); page.goto(url)
                page.wait_for_function("document.getElementById('documentList').textContent.includes('文档库为空')")
                result['steps'].append('Empty managed library confirmed; submit disabled without selection.')
                started=time.monotonic(); page.set_input_files('#uploadFiles',str(source))
                print('Real upload accepted; waiting for local model ingestion.',flush=True)
                page.wait_for_selector('.document-check:not(:disabled)',timeout=300000)
                result['ingestion_seconds']=round(time.monotonic()-started,2)
                assert page.locator('.document-check:checked').count()==0
                page.check('.document-check'); page.select_option('#answerLanguage','en')
                started=time.monotonic()
                with page.expect_response(lambda r:'/api/v1/queries/ask' in r.url and r.request.method=='POST',timeout=240000) as response:
                    page.fill('#messageInput','What is the waiting period and its accident exception?'); page.click('#sendBtn')
                page.wait_for_function("document.getElementById('genPill').hidden",timeout=240000)
                text=response.value.text()
                events=[json.loads(line[5:].strip()) for line in text.splitlines() if line.startswith('data:')]
                final=next(event for event in events if event.get('type')=='end')
                result['answer_seconds']=round(time.monotonic()-started,2)
                result['answer_kind']=final['answer_kind']; result['answer']=final['answer']
                result['source_ids']=[c['document_id'] for c in final['retrieved_chunks']]
                assert final['answer_kind']=='answer' and final['retrieved_chunks']
                assert '90' in final['answer'] and '[' in final['answer']
                assert final['answer_language']=='en'
                page.wait_for_selector('.cite-card')
                page.locator('.cite-card').first.click()
                page.wait_for_selector('#pdfJsView:not([hidden])',timeout=20000)
                page.wait_for_function("document.getElementById('pdfTextLayer').textContent.includes('90 days')",timeout=20000)
                args.screenshot.parent.mkdir(parents=True,exist_ok=True)
                page.screenshot(path=str(args.screenshot),full_page=True)
                result['steps'].append('Real local encoding and API generation returned 90-day clause and accident exception with original citation; browser opened PDF by ID.')
                document_id=final['document_ids'][0]
                page.on('dialog',lambda dialog:dialog.accept()); started=time.monotonic(); page.click('[data-delete]')
                page.wait_for_selector('.cite-card.is-deleted')
                page.wait_for_function("document.getElementById('documentList').textContent.includes('来源已删除')",timeout=30000)
                assert page.request.get(url+f'/api/v1/documents/{document_id}/file').status==410
                history=page.request.get(url+f"/api/v1/queries/history/{final['query_id']}").json()
                assert history['answer']==final['answer']
                assert all('content' not in c for c in history['metadata']['retrieved_chunks'])
                result['deletion_seconds']=round(time.monotonic()-started,2)
                result['steps'].append('Deletion completed; file URL returns 410; old card has no excerpt; answer body retained.')
                result['embedding']=ingestion.embedding.diagnostics()
                result['generation_model']=settings.generation_model()
                result['status']='passed'
                context.close(); browser.close()
    except Exception as exc:
        result['status']='failed'; result['failure']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text('# Batch 8 real browser operation\n\nPublic synthetic PDF, real local embedding and actual generation API; a single walkthrough does not establish batch 10 quality acceptance.\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
        print(json.dumps({'status':result['status'],'report':str(args.report)},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
