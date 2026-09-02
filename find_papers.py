import urllib.request, urllib.parse, xml.etree.ElementTree as ET, time

queries = [
    'id:2204.06806',                                   # YOLO-Pose 原论文（必收）
    'all:"YOLO-Pose"',
    'all:"real-time human pose estimation"',
    'all:"YOLOv8" AND all:"pose estimation"',
    'all:"human pose estimation" AND all:benchmark',
    'all:"OpenPose" OR all:"MoveNet" OR all:"HRNet"',
    'all:"Raspberry Pi" AND all:"deep learning inference"',
    'all:"human pose estimation" AND all:fitness',
    'all:"action quality assessment" OR all:"repetition counting"',
    'all:"edge deployment" AND all:"pose estimation"',
    'all:"NCNN" AND all:"pose estimation"',
    'all:"int8 quantization" AND all:"pose estimation"',
]

ns = {'a': 'http://www.w3.org/2005/Atom'}
seen = set()

def fetch(url):
    last = None
    for _ in range(3):
        try:
            return urllib.request.urlopen(url, timeout=50).read()
        except Exception as e:
            last = e
            time.sleep(2)
    print("ERR_FETCH", url, last)
    return None

for q in queries:
    url = ("http://export.arxiv.org/api/query?search_query=%s&start=0&max_results=8&sortBy=relevance"
           % urllib.parse.quote(q))
    data = fetch(url)
    if not data:
        continue
    try:
        root = ET.fromstring(data)
    except Exception as e:
        print("ERR_PARSE", q, e)
        continue
    for e in root.findall('a:entry', ns):
        idurl = e.find('a:id', ns).text.strip()
        aid = idurl.split('/')[-1]
        if aid in seen:
            continue
        seen.add(aid)
        title = ' '.join(e.find('a:title', ns).text.split())
        summary = ' '.join(e.find('a:summary', ns).text.split())
        authors = [a.find('a:name', ns).text for a in e.findall('a:author', ns)]
        pub = e.find('a:published', ns).text[:10]
        print("====")
        print("ID:", aid)
        print("TITLE:", title)
        print("AUTHORS:", ', '.join(authors[:4]) + (' et al.' if len(authors) > 4 else ''))
        print("DATE:", pub)
        print("LINK:", idurl)
        print("SUMMARY:", summary[:420])
        print()
    time.sleep(1)
print("TOTAL_CANDIDATES:", len(seen))
