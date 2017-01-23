'''
Jennifer Brown
CSE 347
Homework 1
9/15/15
'''
import sys, re


def clean(s):
	'''
	input: a record in original IMDB format
	output: a cleaned record with four fields separated by commas: Year of Production, Country, Title, Title of Episode
	'''    
	title = re.compile("(\")(.*)(\")")
	year = re.compile("(\()([0-9]{4})(\))")
	episode = re.compile("(\{)(.*)(\})")
	country = re.compile("([a-zA-Z]+)$")
	
	t = re.search(title,s)
	y = re.search(year,s)
	e = re.search(episode,s)
	eFinal = re.sub('\(.*\)','',e.group(2))
	c = re.search(country,s)

	return y.group(2)+","+c.group(0)+","+t.group(2)+","+eFinal

if __name__ == '__main__':
    
    for line in sys.stdin:
        print clean(line.strip())


