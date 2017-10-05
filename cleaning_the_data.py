
  '''What is a unicode ? 
  	Unicode is a standard for working with a wide range of characters. Each symbol has a codepoint (a number), and these codepoints can be encoded (converted to a sequence of bytes) using a variety of encodings.
		UTF-8 is one such encoding. The low codepoints are encoded using a single byte, and higher codepoints are encoded as sequences of bytes.
  '''
'''1. Escaping HTML characters: Data obtained from web usually contains a lot of html entities like &lt; &gt; &amp; 
which gets embedded in the original data. It is thus necessary to get rid of these entities.
 One approach is to directly remove them by the use of specific regular expressions. Another approach is to use appropriate 
 packages and modules (for example htmlparser of Python), which can convert these entities to standard html tags.
  For example: &lt; is converted to “<” and &amp; is converted to “&”.'''

import HTMLParser

original_tweet = "I luv my &lt;3 iphone &amp; you’re awsm apple. DisplayIsAwesome, sooo happppppy http://www.apple.com"
# converting it into unicode:

original_tweet = unicode(original_tweet,'utf-8')
html_parser = HTMLParser.HTMLParser()
tweet = html_parser.unescape(original_tweet)
# now tweet = ->

tweet = "I luv my <3 iphone & you’re awsm apple. DisplayIsAwesome, sooo happppppy 🙂 http://www.apple.com"


 2. Decoding data: Thisis the process of transforming information from complex symbols to simple
  and easier to understand characters. Text data may be subject to different forms of decoding 
  like “Latin”, “UTF8” etc. Therefore, for better analysis, it is necessary to keep the complete
   data in standard encoding format. UTF-8 encoding is widely accepted and is recommended to use.
   eg;->
		tweet = original_tweet.decode("utf8").encode(‘ascii’,’ignore’)
Output:
     "I luv my <3 iphone & you’re awsm apple. DisplayIsAwesome, sooo happppppy 🙂 http://www.apple.com"



 Apostrophe Lookup: To avoid any word sense disambiguation in text, it is recommended to maintain proper 
 structure in it and to abide by the rules of context free grammar. When apostrophes are used, chances of disambiguation increases.
For example “it’s is a contraction for it is or it has”.

All the apostrophes should be converted into standard lexicons. One can use a lookup table of all possible keys to get rid of disambiguates.

Snippet:

APPOSTOPHES = {“'s" : " is", "'re" : " are", ...} ## Need a huge dictionary"

words = tweet.split()

reformed = [APPOSTOPHES[word] if word in APPOSTOPHES else word for word in words]

reformed = " ".join(reformed)







4.Removal of Stop-words: When data analysis needs to be data driven at the word level, the commonly occurring words (stop-words) should be removed. One can either create a long list of stop-words or one can use predefined language specific libraries.
5.Removal of Punctuations: All the punctuation marks according to the priorities should be dealt with. For example: “.”, “,”,”?” are important punctuations that should be retained while others need to be removed.
6.Removal of Expressions: Textual data (usually speech transcripts) may contain human expressions like [laughing], [Crying], [Audience paused]. These expressions are usually non relevant to content of the speech and hence need to be removed. Simple regular expression can be useful in this case.

7.Split Attached Words: We humans in the social forums generate text data, which is completely informal in nature. Most of the tweets are accompanied with multiple attached words like RainyDay, PlayingInTheCold etc. These entities can be split into their normal forms using simple rules and regex.
8.Slangs lookup: Again, social media comprises of a majority of slang words. These words should be transformed into standard words to make free text. The words like luv will be converted to love, Helo to Hello. The similar approach of apostrophe look up can be used to convert slangs to standard words. A number of sources are available on the web, which provides lists of all possible slangs, this would be your holy grail and you could use them as lookup dictionaries for conversion purposes.

Snippet:

            tweet = _slang_loopup(tweet)

Outcome:

>>  “I love my <3 iphone & you are awesome apple. Display Is Awesome, sooo happppppy 🙂 http://www.apple.com”


   9. Standardizing words: Sometimes words are not in proper formats. For example: “I looooveee you” should be “I love you”. Simple rules and regular expressions can help solve these cases.

Snippet:

tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))

Outcome:

>>  “I love my <3 iphone & you are awesome apple. Display Is Awesome, so happy 🙂 http://www.apple.com”



10. 
    Removal of URLs: URLs and hyperlinks in text data like comments, reviews, and tweets should be removed.

 

Final cleaned tweet:

>>  “I love my iphone & you are awesome apple. Display Is Awesome, so happy!” , <3 , 🙂

 


 Advanced data cleaning:

    Grammar checking: Grammar checking is majorly learning based, huge amount of proper text data is learned and models are created for the purpose of grammar correction. There are many online tools that are available for grammar correction purposes.
   
    Spelling correction: In natural language, misspelled errors are encountered. Companies like Google and Microsoft have achieved a decent accuracy level in automated spell correction. One can use algorithms like the Levenshtein Distances, Dictionary Lookup etc. or other modules and packages to fix these errors.

