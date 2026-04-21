from crawler.search_api import get_top_institutions_for_region

insts = get_top_institutions_for_region("Punjab, India")
print("FOUND:", insts)
