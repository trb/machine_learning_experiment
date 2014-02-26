#Machine learning experiments

The idea of this experiment is to classify apartment postings on craigslist ones I would like, and ones I would dislike.

This would allow me to be notified any time a new apartment was posted that I would like, so I could contact the landlord earlier than anyone else.

More importantly, it was a fun attempt at using Scikit-learn and different machine learning algorithms.

#Structure

scraper.py will download links to apartments posted on Craigslist. These are put into the "likes"-array in the get_train_data.html file.

Then, opening up get_train_data.html will show all the apartment postings and allow you to classify the posting as "Like" or "Dislike". Once down, "Get results" will post the liked/disliked links to the developer tools (with console.log).

Put the liked/disliked links into data/like and data/dislike, respectivly.

Experiment.py will then download all the posts, and train a machine learning algorithm with a subset of the posts using the like/dislike classification. Then it evaluated the algorithm by predicting the category of the remaining posts and verifying the prediction with the like/dislike classification. The best algorithm for me was ExtraTreesClassifier with 76% correct predictions.

Features used are: price, size (sqft), location (lat/lng) and number of images.

#Ideas

It'd be cool to look at the histogram of the posted images. Professional images (and therefore better postings) usually have a better exposure leading to a more ideal histogram. Bad postings often have picture taken with a cell phone or cheap camera, leading to unbalanced histograms. This could be a powerful feature.
