"""Real downloaded pretrained weights; no mock inference in this test."""
from visioninspect import Encoder, analyze, sample

def test_pretrained_pipeline():
    r,t,l=sample()
    result=analyze(r,t,Encoder(),l)
    assert result['bank_patches']==2048
    assert result['results'][0]['patch_shape']==[27,27]
    assert all(0<=row['score']<=2 for row in result['results'])
    assert 0<=result['metrics']['auroc']<=1
