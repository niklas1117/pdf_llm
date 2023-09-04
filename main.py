from pathlib import Path
import os
from pdf_llm import (PdfLLM, PdfLLMStuff)

paths = [Path().cwd().joinpath('factor_momentum_papers').joinpath(paper) for paper in os.listdir('factor_momentum_papers')]

os.makedirs('fmom_summaries', exist_ok=True)

# gpt 4 very expensive & doesn't work with stuff 
# only with refine (even more expensive)

# pdfllmstuff = PdfLLMStuff(str(path), model_name='gpt-4')
names = [
    'Factor Momentum and the Momentum Factor',
    'Scaling up Market Anomalies',
    'Factor Momentum',
    'Factor Momentum Everywhere',
    'Reexamination of Factor Momentum'
]

for path, name in zip(paths[:1], names[:1]):
    #gpt-3 is cheaper
    pdfllm = PdfLLM(str(path), model_name='gpt-3.5-turbo')
    pdfllm.save_latex_summary(name, str(Path('fmom_summaries').joinpath('gpt_3'+path.name.split('.')[0]+'_summary')))
