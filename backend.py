from flask import Flask
from TriMention.web import parse_mentions
from MCMention.applyMention import topkMention
from MCSubtitle.applyMCSubtitle import topkSubTitle,topkSubTitleWithScores
from MCRelation.applyRelation import topkRelation

import json

app = Flask(__name__)
app.config['JSON_AS_ASCII'] =False
@app.route("/readyToUse/<string:qs>")
def useAllFeatures(qs):
	mentions = parse_mentions(qs)
	mentionDetection = topkMention(qs,mentions,K=3)
	disambiguationBySubTitle = topkSubTitle(qs,mentionDetection,K=3)
	disambiguationByRelation = list(topkRelation(qs,disambiguationBySubTitle,K=1)[0])
	disambiguationByRelation[2] = float(disambiguationByRelation[2])
	return {"data":disambiguationByRelation}

app.run(host="localhost",port=9899)
