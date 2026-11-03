import pytest
from test_scoped_query_api import scoped, ready, ask  # noqa: F401
from test_document_api import api  # noqa: F401
from test_index_coordinator import library  # noqa: F401


def select(scoped,ids):
    return scoped.client.post('/api/v1/queries/session',json={'session_id':'selection-test','document_ids':ids})


def test_empty_selection_advances_epoch_without_enabling_global_query(scoped):
    doc=ready(scoped)
    initial=select(scoped,[doc,doc]).json()
    assert initial['document_ids']==[doc]
    first=scoped.client.post('/api/v1/queries/ask',json={'question':'Waiting period?', 'document_ids':[doc], 'session_id':'selection-test'}).json()
    assert first['scope_epoch']==initial['scope_epoch']
    cleared=select(scoped,[]).json()
    assert cleared['scope_epoch']>initial['scope_epoch']
    restored=select(scoped,[doc]).json()
    assert restored['scope_epoch']>cleared['scope_epoch']
    follow=scoped.client.post('/api/v1/queries/ask',json={'question':'What about accidents?', 'document_ids':[doc], 'session_id':'selection-test'}).json()
    assert follow['answer_kind']=='clarification'
    assert ask(scoped,[]).status_code==422


@pytest.mark.parametrize('ids',[None,[999],['1'],[True]])
def test_invalid_selection_is_rejected_before_context_change(scoped,ids):
    assert select(scoped,ids).status_code==422


def test_delayed_old_request_is_bound_to_its_prepared_epoch(scoped):
    doc=ready(scoped)
    old=select(scoped,[doc]).json()
    select(scoped,[])
    latest=select(scoped,[doc]).json()
    response=scoped.client.post('/api/v1/queries/ask',json={'question':'Waiting period?', 'document_ids':[doc], 'session_id':'selection-test', 'scope_epoch':old['scope_epoch']})
    assert response.status_code==200 and response.json()['scope_epoch']==old['scope_epoch']
    follow=scoped.client.post('/api/v1/queries/ask',json={'question':'What about accidents?', 'document_ids':[doc], 'session_id':'selection-test', 'scope_epoch':latest['scope_epoch']}).json()
    assert follow['answer_kind']=='clarification'
    assert select(scoped,[doc]).json()['scope_epoch']==latest['scope_epoch']
    assert scoped.client.post('/api/v1/queries/ask',json={'question':'Q', 'document_ids':[doc], 'session_id':'selection-test', 'scope_epoch':999}).status_code==422
