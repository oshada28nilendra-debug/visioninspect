import io
import numpy as np
import pytest
from PIL import Image
from visioninspect import analyze, create_app, decode, distances, png, sample

def fake(im):
    return np.asarray(im.resize((4,4)),dtype=np.float32)+1

def test_distance_known_and_zero():
    np.testing.assert_allclose(distances([[1,0],[0,1],[0,0]],[[1,0]]),[0,2**.5,1],atol=1e-6)

@pytest.mark.parametrize('x', [[], [[float('nan'),1]], [[float('inf'),1]]])
def test_invalid_features(x):
    with pytest.raises(ValueError):distances(x,[[1,0]])

def test_dimensions():
    with pytest.raises(ValueError):distances([[1,2]],[[1,2,3]])

@pytest.mark.parametrize('data',[b'bad',b'x'*(5*1024*1024+1),png(Image.new('RGB',(20,20)))])
def test_bad_images(data):
    with pytest.raises(ValueError):decode(data)

def test_gif_rejected():
    out=io.BytesIO();Image.new('RGB',(64,64)).save(out,format='GIF')
    with pytest.raises(ValueError):decode(out.getvalue())

def test_sample_reproducible_disjoint():
    r,t,l=sample();assert (r,t,l)==sample()
    assert not set(data for _,data in r)&set(data for _,data in t)
    assert sum(l.values())==12 and len(r)==12

def test_ranking_export_and_metrics():
    r,t,l=sample();result=analyze(r,t,fake,l,True)
    assert result['metrics'] is not None
    assert len(result['results'])==24
    scores=[x['score'] for x in result['results']];assert scores==sorted(scores,reverse=True)
    assert result['results'][0]['overlay'].startswith('data:image/png;base64,')
    assert analyze(r,t[:2],fake,{n:0 for n,_ in t[:2]})['metrics'] is None

@pytest.mark.parametrize('case',['overlap','duplicate_ref','duplicate_eval','bad_labels','bad_count','same_name'])
def test_invalid_sets(case):
    r,t,l=sample();labels=None
    if case=='overlap':t=[r[0]]
    if case=='duplicate_ref':r=[r[0],r[0]]
    if case=='duplicate_eval':t=[t[0],('renamed.png',t[0][1])]
    if case=='bad_labels':labels={n:True for n,_ in t}
    if case=='bad_count':r=r[:1]
    if case=='same_name':t=[('a',t[0][1]),('a',t[1][1])]
    with pytest.raises(ValueError):analyze(r,t,fake,labels)

def test_no_evaluation_in_bank():
    r,t,_=sample();seen=[]
    def spy(im):seen.append(im.tobytes());return fake(im)
    one=analyze(r,t[:1],spy)['results'][0]['score']
    all_rows=analyze(r,t,spy)['results']
    assert next(x['score'] for x in all_rows if x['name']==t[0][0])==one

def test_web_sample_and_errors():
    c=create_app(fake).test_client()
    assert c.get('/').status_code==200
    assert c.post('/analyze').status_code==400
    assert c.post('/analyze',data={'labels':'{' }).status_code==400
    response=c.post('/analyze',data={'sample':'true'})
    assert response.status_code==200 and response.json['synthetic_sample']
    assert 'frame-ancestors' in response.headers['Content-Security-Policy']

def test_model_download_failure(monkeypatch):
    def fail(): raise OSError('offline')
    monkeypatch.setattr('visioninspect.Encoder',fail)
    response=create_app().test_client().post('/analyze',data={'sample':'true'})
    assert response.status_code==503
    assert 'Pretrained model unavailable' in response.json['error']
