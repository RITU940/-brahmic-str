import urllib.request, xml.etree.ElementTree as ET, re

IDS = {
 "indicstr12":"2403.08007","bstd":"2511.23071","glotocr":"2604.12978","mgpstr":"2307.13244",
 "gradet":"2509.18081","beyondfertility":"2510.09947","hiercode":"2403.13761","strokezs":"2106.11613",
 "star":"2210.08490","moose":"2407.18616","openset1":"2204.05535","openset2":"2203.05179",
 "unseendet":"2307.15991","maestrou":"2210.10027","crosslingstr":"2312.10806","strscaling":"2401.00028",
 "taskanalogies":"2604.09713","unionst":"2602.06450","manchu":"2507.06761","khmerst":"2410.18277",
 "univkhmer":"2603.00702","qwen25vl":"2502.13923","florence2":"2311.06242","parseq":"2207.06966",
 "subwordoverlap":"2310.10378","overlap2":"2405.12413","preregml":"2405.02200","repro":"2311.18807",
}
ns={"a":"http://www.w3.org/2005/Atom"}
out=[]
ids=list(IDS.items())
url="http://export.arxiv.org/api/query?id_list="+",".join(v for _,v in ids)+"&max_results=100"
xml=urllib.request.urlopen(url,timeout=60).read()
root=ET.fromstring(xml)
by_id={}
for e in root.findall("a:entry",ns):
    aid=e.find("a:id",ns).text
    m=re.search(r"abs/([0-9.]+)",aid)
    if m: by_id[m.group(1)]=e
for key,aid in ids:
    e=by_id.get(aid)
    if e is None:
        out.append(f"%% MISSING from arXiv API: {key} {aid}\n"); continue
    title=re.sub(r"\s+"," ",e.find("a:title",ns).text.strip())
    authors=[a.find("a:name",ns).text for a in e.findall("a:author",ns)]
    year=e.find("a:published",ns).text[:4]
    out.append("@article{%s,\n  author={%s},\n  title={%s},\n  journal={arXiv preprint arXiv:%s},\n  year={%s}\n}\n"%(key," and ".join(authors),title,aid,year))
open("main.bib","w").write("\n".join(out))
print("wrote", sum(1 for o in out if o.startswith('@')), "entries;", sum(1 for o in out if o.startswith('%%')), "missing")
