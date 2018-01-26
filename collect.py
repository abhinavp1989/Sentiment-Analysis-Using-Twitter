"""
collect.py
"""
from collections import Counter
from TwitterAPI import TwitterAPI
import pickle
import sys
import time

'''Access Tokes for Twitter API'''

consumer_key = '------------------------'
consumer_secret = '----------------------------------'
access_token = '----------------------------------------------'
access_token_secret = '-------------------------------------------'

def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)
            
def request(twitter):
    """Collect training and test data for classification 
    and build data for clustering
    """
    tweets = []
    test_tweets =[]
    limit_tweets = 1500 # limit for training and testing data
    
    for r in range(0,20):
        req = robust_request(twitter, 'search/tweets', {'q': 'exam', 'count': 100,"lang":'en'})
        for i in req:
            if len(tweets) < limit_tweets:
                tweets.append(i) #Used for training
            else:
                test_tweets.append(i) #used for testing
    #print(len(tweets))
    
    
    for i in tweets:
        pickle.dump(tweets, open('data.pkl', 'wb')) #Train Data
        
    for i in test_tweets:
        pickle.dump(test_tweets, open('test_data.pkl', 'wb')) #Test Data
        
    f = open("cluster_data.txt", "w+") #Used for cluster 
    name=[]
    for t in tweets:
        name.append(t['user']['screen_name'])
    for s in set(name[:15]):
        request=robust_request(twitter,'followers/ids', {'screen_name': s,'count': 50})
        for r in request:
            f.write("%s,%s\n" %(s,r))
    f.close()
            
    f = open('tweet_data.txt', 'w+')
    f.write("NUMBER OF USERS COLLECTED : %d\n" % (len(set(name))))
    f.write("NUMBER OF MESSAGES COLLECTED in total: %d\n" % (len(tweets)+len(test_tweets)))
    f.write("NUMBER OF MESSAGES COLLECTED for training model : %d\n" % (len(tweets)))
    f.write("NUMBER OF MESSAGES COLLECTED for testing model: %d\n" % (len(test_tweets)))
    f.close()
        
def main():
   '''Main Method'''
   twitter = get_twitter()
   request(twitter)
    
if __name__ == '__main__':
    main()
