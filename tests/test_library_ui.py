"""Real Chromium interactions; explicit model substitutes establish UI contracts only."""
import asyncio
import json
import threading
import pytest
from pathlib import Path
from test_pdf_quality import make_pdf
from ingestion_fakes import LocalEncoderFake
from index_fakes import VectorStoreFake
from test_scoped_query_api import QueryEncoder
from app.services.scoped_query import ScopedQueryProcessor
from ui_harness import serve_library

playwright = pytest.importorskip('playwright.sync_api')

class BrowserGenerator:
    calls=[]
    started=threading.Event()
    def has_credentials(self): return True
    def answer(self,chunks):
        if self.answer_language=='en':
            return '[Conclusion]\nSee [1].\n[Clause evidence]\nSource conditions [1].\n[Uncertainty / boundaries]\nNot a claim promise.'
        return '【结论】\n见 [1]。\n【条款依据】\n条款条件 [1]。\n【不确定/边界】\n不构成理赔承诺。'
    async def agenerate_response(self,q,chunks):
        self.calls.append((q,list(chunks)))
        return {'answer':self.answer(chunks),'success':True}
    async def agenerate_stream(self,q,chunks):
        self.calls.append((q,list(chunks)))
        if 'slow' in q:
            self.started.set()
            yield 'Partial answer'
            await asyncio.sleep(30)
        yield self.answer(chunks)

@pytest.fixture
def ui(tmp_path):
    BrowserGenerator.calls=[]; BrowserGenerator.started.clear()
    with serve_library(tmp_path/'library',embedding_factory=LocalEncoderFake,
                       processor=ScopedQueryProcessor(QueryEncoder,BrowserGenerator),vectors=VectorStoreFake()) as (url,ingestion,deletion):
        with playwright.sync_playwright() as p:
            try: browser=p.chromium.launch(channel='msedge',headless=True)
            except playwright.Error:
                try: browser=p.chromium.launch(headless=True)
                except playwright.Error: pytest.skip('Install Playwright Chromium or Microsoft Edge for browser tests')
            context=browser.new_context(viewport={'width':1440,'height':1000})
            page=context.new_page()
            errors=[]; page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(url)
            page.wait_for_function("document.getElementById('documentList').textContent.includes('文档库为空')")
            yield page,tmp_path,ingestion,deletion,url
            assert not errors,errors
            context.close(); browser.close()


def upload(page,root,name='policy.pdf',text='Waiting period is 90 days, except accidents.',pages=None):
    path=make_pdf(root/name,pages or [text])
    page.set_input_files('#uploadFiles',str(path))
    page.wait_for_function("!document.getElementById('uploadFiles').disabled")
    page.wait_for_function("Array.from(document.querySelectorAll('.document-check')).some(c=>!c.disabled)")
    return path


def send(page,q):
    page.fill('#messageInput',q); page.click('#sendBtn')
    page.wait_for_function("document.getElementById('genPill').hidden",timeout=20000)


def test_upload_selection_persistence_empty_scope_and_pdf_independence(ui):
    page,root,_,_,_=ui
    assert page.is_checked('#streamToggle')
    page.fill('#messageInput','Waiting period?')
    assert page.is_disabled('#sendBtn')
    upload(page,root)
    assert not page.is_checked('.document-check')
    page.check('.document-check'); page.click('[data-preview]')
    assert page.is_checked('.document-check')
    page.select_option('#answerLanguage','en')
    send(page,'What is the waiting period?')
    assert page.locator('.answer-section-label').first.text_content()=='Conclusion'
    assert page.locator('.cite-card').count()>0
    upload(page,root,'second.pdf','Waiting period is 180 days.')
    assert page.locator('.document-check:checked').count()==1
    page.reload(); page.wait_for_selector('.document-check:checked')
    assert page.locator('.document-check:checked').count()==1
    page.click('#clearSelection')
    assert page.locator('.document-check:checked').count()==0
    page.reload(); page.wait_for_selector('.document-check')
    assert page.locator('.document-check:checked').count()==0
    page.fill('#messageInput','Waiting period?'); assert page.is_disabled('#sendBtn')


def test_same_name_id_preview_deletion_and_html_escape(ui):
    page,root,_,deletion,url=ui
    first=upload(page,root,'same.pdf')
    page.check('.document-check')
    send(page,'Waiting period?')
    # Same display name, different original bytes and stable identities.
    second=make_pdf(root/'different.pdf',['Waiting period is 180 days.'])
    page.set_input_files('#uploadFiles',{'name':'same.pdf','mimeType':'application/pdf','buffer':second.read_bytes()})
    page.wait_for_function("document.querySelectorAll('.document-check:not(:disabled)').length===2")
    page.locator('[data-preview]').nth(1).click()
    page.wait_for_function("document.getElementById('pdfSelect').value==='2'")
    labels=page.locator('#pdfSelect option').all_text_contents()
    assert len(set(labels))==2 and all('#' in label for label in labels)
    assert '#1' in page.locator('.cite-card').first.inner_text()
    assert page.locator('.document-check:checked').first.get_attribute('data-id')=='1'
    page.on('dialog',lambda dialog:dialog.accept())
    page.locator('[data-delete]').first.click()
    page.wait_for_selector('.cite-card.is-deleted')
    assert '来源已删除' in page.locator('.cite-card').first.inner_text()
    assert 'Waiting period' not in page.locator('.cite-card').first.inner_text()
    assert page.locator('.cite-card').first.is_disabled()
    assert page.locator('.document-check:checked').count()==0
    assert page.request.get(url+'/api/v1/documents/1/file').status==410
    page.wait_for_function("document.getElementById('pdfSelect').options.length===1 && document.getElementById('pdfSelect').value==='2'")
    unsafe='<img src=x onerror=alert(1)>.pdf'
    page.set_input_files('#uploadFiles',{'name':unsafe,'mimeType':'application/pdf','buffer':make_pdf(root/'third.pdf',['No cover for war.']).read_bytes()})
    page.wait_for_function("document.querySelectorAll('.document-row').length===3")
    assert page.locator('.document-row img').count()==0
    assert '<img' in page.locator('.document-row').last.inner_text()


def test_scope_changes_create_epochs_without_queries_and_modes(ui):
    page,root,_,_,url=ui
    upload(page,root); upload(page,root,'b.pdf','Waiting period is 180 days.')
    page.locator('.document-check').first.check()
    send(page,'Waiting period?')
    first=page.locator('.message-provenance').last.inner_text()
    page.locator('.document-check').nth(1).check()
    page.locator('.document-check').nth(1).uncheck()
    send(page,'What about accidents?')
    assert page.locator('.message-provenance').last.inner_text()!=first
    assert '请澄清' in page.locator('.message.is-assistant').last.inner_text()
    page.locator('.document-check').nth(1).check()
    send(page,'Compare waiting periods')
    assert '逐文档比较' in page.locator('#modeStatus').inner_text()
    page.click('#resetBtn')
    send(page,'这些期限呢？')
    assert '请澄清' in page.locator('.message.is-assistant').last.inner_text()


def test_scope_snapshot_is_frozen_during_stream_and_deletion_interrupts(ui):
    page,root,_,_,_=ui
    upload(page,root); upload(page,root,'b.pdf','Waiting period is 180 days.')
    page.locator('.document-check').first.check()
    page.fill('#messageInput','Waiting period slow'); page.click('#sendBtn')
    page.wait_for_function("document.querySelector('.message.is-assistant').textContent.includes('Partial')")
    page.locator('.document-check').first.uncheck(); page.locator('.document-check').nth(1).check()
    assert '1' in str(BrowserGenerator.calls[0][1][0]['document_id'])
    page.on('dialog',lambda dialog:dialog.accept())
    page.locator('[data-delete]').first.click()
    page.wait_for_function("document.getElementById('genPill').hidden")
    assert '来源删除' in page.locator('.message.is-assistant').last.inner_text()
    assert page.locator('.cite-card').count()==0
    assert page.locator('.document-check:checked').first.get_attribute('data-id')=='2'


def test_page_limit_and_capacity_advisory(ui):
    page,root,ingestion,_,_=ui
    path=make_pdf(root/'too-many.pdf',['Waiting period is 90 days.']*201)
    page.set_input_files('#uploadFiles',str(path))
    page.wait_for_function("document.getElementById('documentList').textContent.includes('200 页上限')",timeout=20000)
    assert page.is_disabled('.document-check')
    from sqlalchemy.orm import Session
    from app.models.database_models import Document
    with Session(ingestion.engine) as db:
        for i in range(50): db.add(Document(filename=f'pending-{i}.pdf',processing_status='pending'))
        db.commit()
    page.wait_for_function("document.getElementById('libraryNotice').textContent.includes('50')")
    assert not page.is_disabled('#uploadFiles')



def test_multi_upload_and_comparison_limit_does_not_limit_ordinary_qa(ui):
    page,root,_,_,_=ui
    paths=[make_pdf(root/f'{i}.pdf',[f'Policy {i}: waiting period is {90+i} days.']) for i in range(6)]
    page.set_input_files('#uploadFiles',[str(path) for path in paths])
    page.wait_for_function("document.querySelectorAll('.document-check:not(:disabled)').length===6",timeout=30000)
    assert page.locator('.document-check:checked').count()==0
    for checkbox in page.locator('.document-check').all(): checkbox.check()
    page.select_option('#queryMode','qa')
    send(page,'Explain waiting periods')
    assert page.locator('.message.is-assistant').count()==1
    page.select_option('#queryMode','compare')
    page.fill('#messageInput','Compare waiting periods'); page.click('#sendBtn')
    assert '5' in page.locator('#errPill').inner_text()
    assert page.locator('.message.is-assistant').count()==1
    page.select_option('#queryMode','auto')
    send(page,'Compare waiting periods')
    assert 'comparison_limit' in page.locator('#errPill').inner_text()
    assert len(BrowserGenerator.calls)==1


def test_deletion_failure_shows_retry_and_eventually_clears_pdf(ui):
    page,root,_,deletion,_=ui
    upload(page,root); page.check('.document-check'); send(page,'Waiting period?')
    page.wait_for_function("document.getElementById('pdfSelect').value==='1'")
    deletion.fault=lambda stage: (_ for _ in ()).throw(OSError('failure')) if stage=='original' else None
    page.on('dialog',lambda dialog:dialog.accept()); page.click('[data-delete]')
    page.wait_for_function("document.getElementById('documentList').textContent.includes('清理未完成')",timeout=10000)
    assert page.locator('[data-delete]').inner_text()=='重试删除'
    assert page.locator('#pdfJsView').is_hidden() and page.locator('#pdfFrame').is_hidden()
    assert page.locator('#evidenceStrip').is_hidden()
    deletion.fault=lambda stage:None
    page.click('[data-delete]')
    page.wait_for_function("document.getElementById('documentList').textContent.includes('来源已删除')",timeout=10000)


def test_stream_without_terminal_is_not_shown_as_completed(ui):
    page,root,_,_,_=ui
    upload(page,root); page.check('.document-check')
    page.route('**/api/v1/queries/ask',lambda route:route.fulfill(status=200,content_type='text/event-stream',body='event: token\ndata: {"type":"token","content":"Partial output"}\n\n'))
    send(page,'Waiting period?')
    assert '未完成' in page.locator('.message.is-assistant').last.inner_text()
    assert page.locator('.cite-card').count()==0



def test_citation_page_jump_renders_correct_original_page(ui):
    page,root,_,_,_=ui
    upload(page,root,pages=['Policy conditions on page one.','Waiting period is 90 days on page two.'])
    page.check('.document-check'); send(page,'Waiting period?')
    page.locator('.cite-card').filter(has_text='p.2').first.click()
    page.wait_for_selector('#pdfJsView:not([hidden])')
    page.wait_for_function("document.getElementById('pageBadge').textContent==='p.2' && document.getElementById('pdfTextLayer').textContent.includes('page two')")
    assert page.locator('.document-check').is_checked()



def test_delayed_http_question_cannot_populate_returned_scope(ui):
    page,root,_,_,url=ui
    upload(page,root)
    page.check('.document-check')
    held=[]
    def delay(route): held.append(route)
    page.route('**/api/v1/queries/ask',delay)
    page.fill('#messageInput','Waiting period?'); page.click('#sendBtn')
    page.wait_for_timeout(250)
    assert held
    old_epoch=held[0].request.post_data_json['scope_epoch']
    with page.expect_response(lambda r:'/api/v1/queries/session' in r.url and r.request.post_data_json['document_ids']==[]) as cleared:
        page.click('#clearSelection')
    cleared.value.json()
    with page.expect_response(lambda r:'/api/v1/queries/session' in r.url and r.request.post_data_json['document_ids']==[1]) as restored:
        page.check('.document-check')
    new_epoch=restored.value.json()['scope_epoch']
    assert new_epoch>old_epoch
    held[0].continue_()
    page.wait_for_function("document.getElementById('genPill').hidden")
    assert 'epoch '+str(old_epoch) in page.locator('.message-provenance').last.inner_text()
    page.unroute('**/api/v1/queries/ask',delay)
    send(page,'What about accidents?')
    assert '请澄清' in page.locator('.message.is-assistant').last.inner_text()


def test_deleted_pdf_cannot_return_after_async_old_document_destruction(ui):
    page,root,_,_,_=ui
    upload(page,root); upload(page,root,'b.pdf','Waiting period is 180 days.')
    page.evaluate("""() => {
      window.pendingDestroy=[];
      window.releaseDestroyed=()=>window.pendingDestroy.splice(0).forEach(resolve=>resolve());
      window.pdfjsLib={...window.pdfjsLib, getDocument:({url})=>({promise:Promise.resolve({
        numPages:1,
        destroy:()=>url.includes('/1/') ? new Promise(resolve=>{window.pendingDestroy.push(resolve);}) : Promise.resolve(),
        getPage:async()=>({getViewport:()=>({width:100,height:100}),
          render:()=>({promise:Promise.resolve(),cancel:()=>{}}),getTextContent:async()=>({items:[]})})
      })})};
    }""")
    page.locator('[data-preview]').first.click()
    page.wait_for_selector('#pdfJsView:not([hidden])')
    page.locator('[data-preview]').nth(1).click()
    page.wait_for_function('window.pendingDestroy.length>0')
    page.on('dialog',lambda dialog:dialog.accept())
    page.locator('[data-delete]').nth(1).click()
    page.wait_for_function("document.getElementById('pdfJsView').hidden && document.getElementById('pdfFrame').hidden")
    page.evaluate('window.releaseDestroyed()')
    page.wait_for_timeout(200)
    assert page.locator('#pdfJsView').is_hidden() and page.locator('#pdfFrame').is_hidden()
    assert page.locator('#evidenceStrip').is_hidden()



def test_explicit_clear_persists_when_saved_document_is_temporarily_unready(ui):
    page,root,ingestion,_,_=ui
    upload(page,root); page.check('.document-check')
    from sqlalchemy.orm import Session
    from app.models.database_models import Document
    with Session(ingestion.engine) as db:
        db.get(Document,1).processing_status='pending'; db.commit()
    page.reload(); page.wait_for_selector('.document-check:disabled')
    page.click('#clearSelection')
    assert page.evaluate("localStorage.getItem('insur_document_ids')")=='[]'
    with Session(ingestion.engine) as db:
        db.get(Document,1).processing_status='ready'; db.commit()
    page.reload(); page.wait_for_selector('.document-check:not(:disabled)')
    assert not page.is_checked('.document-check')
