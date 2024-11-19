import time
from colorama import init, Fore
init(autoreset=True)

from easy_text_clustering.clusterer import ClusterClassifier

failures = 0

cc = ClusterClassifier()

TEXTS = sample_strings = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "Time flies when you're having fun.",
    "Better late than never.",
    "Actions speak louder than words.",
    "The early bird catches the worm.",
    "A picture is worth a thousand words.",
    "When in Rome, do as the Romans do.",
    "The pen is mightier than the sword.",
    "Practice makes perfect.",
    "It’s always darkest before the dawn.",
    "Don’t count your chickens before they hatch.",
    "Necessity is the mother of invention.",
    "You can’t have your cake and eat it too.",
    "Too many cooks spoil the broth.",
    "Rome wasn’t built in a day.",
    "An apple a day keeps the doctor away.",
    "Every cloud has a silver lining.",
    "A watched pot never boils.",
    "If you want something done right, do it yourself.",
    "Laughter is the best medicine.",
    "The grass is always greener on the other side.",
    "Money doesn’t grow on trees.",
    "There’s no place like home.",
    "Don’t bite the hand that feeds you.",
    "Out of sight, out of mind.",
    "What goes around comes around.",
    "Haste makes waste.",
    "Barking up the wrong tree.",
    "A bird in the hand is worth two in the bush.",
    "Don’t put all your eggs in one basket.",
    "The squeaky wheel gets the grease.",
    "You can lead a horse to water, but you can’t make it drink.",
    "Beauty is in the eye of the beholder.",
    "Good things come to those who wait.",
    "Strike while the iron is hot.",
    "A fool and his money are soon parted.",
    "You can’t make an omelette without breaking eggs.",
    "There’s no time like the present.",
    "Blood is thicker than water.",
    "A chain is only as strong as its weakest link.",
    "Don’t judge a book by its cover.",
    "Knowledge is power.",
    "Absence makes the heart grow fonder.",
    "The truth will set you free.",
    "If it ain’t broke, don’t fix it.",
    "The best things in life are free.",
    "Money can’t buy happiness.",
    "It’s not the destination, it’s the journey.",
    "You can’t teach an old dog new tricks.",
    "A stitch in time saves nine.",
    "The apple doesn’t fall far from the tree.",
    "You win some, you lose some.",
    "What doesn’t kill you makes you stronger.",
    "Jack of all trades, master of none.",
    "There’s no use crying over spilled milk.",
    "Hindsight is 20/20.",
    "One man’s trash is another man’s treasure.",
    "A rolling stone gathers no moss.",
    "Don’t throw the baby out with the bathwater.",
    "Beauty is only skin deep.",
    "Curiosity killed the cat.",
    "An ounce of prevention is worth a pound of cure.",
    "Don’t let the bed bugs bite.",
    "It takes two to tango.",
    "You can’t judge a book by its cover.",
    "No news is good news.",
    "A penny saved is a penny earned.",
    "The road to hell is paved with good intentions.",
    "If the shoe fits, wear it.",
    "The customer is always right.",
    "Two heads are better than one.",
    "You can’t please everyone.",
    "What you see is what you get.",
    "He who hesitates is lost.",
    "Don’t sweat the small stuff.",
    "Sometimes the best things come in small packages.",
    "Good things come in threes.",
    "Life is what happens when you’re busy making other plans.",
    "All good things must come to an end.",
    "You can’t always get what you want.",
    "Don’t make a mountain out of a molehill.",
    "A penny for your thoughts.",
    "Make hay while the sun shines.",
    "The early bird gets the worm.",
    "If you can’t beat them, join them.",
    "Variety is the spice of life.",
    "You are what you eat.",
    "The bigger they are, the harder they fall.",
    "If you don’t have anything nice to say, don’t say anything at all.",
    "The grass is always greener where you water it.",
    "An eye for an eye makes the whole world blind.",
    "You can’t put a square peg in a round hole.",
    "The whole is greater than the sum of its parts.",
    "What’s done is done.",
    "Let sleeping dogs lie.",
    "It’s better to give than to receive.",
    "There’s no place like home.",
    "The devil is in the details.",
    "Silence is golden.",
    "Out of the frying pan and into the fire.",
    "A leopard can’t change its spots.",
    "Don’t cross that bridge until you come to it.",
    "Better safe than sorry.",
    "It’s never too late to start.",
    "You can’t please everyone.",
    "A fool's paradise is the best place to be.",
    "That’s the way the cookie crumbles.",
    "Don’t burn your bridges.",
    "Life’s too short to waste time."
]

def readme_basic_fit():
    # run the pipeline:
    embs, labels, summaries = cc.fit(TEXTS)
    # show the results
    cc.show()
    # save 
    cc.save("temp/cc_100k")

def readme_basic_inference():
    cc = ClusterClassifier()
    # load state
    cc.load("temp/cc_100k")
    # visualize
    cc.show()
    # classify new texts with k-nearest neighbour search
    cluster_labels, embeddings = cc.infer(TEXTS, top_k=1)

def readme_advanced():
    # initialize the ClusterClassifier to use TruncatedSVD with appropriate params
    # also set the clustering to use KMeans clustering with appropriate params
    cc = ClusterClassifier(
        projection_algorithm='tsvd', 
        projection_args={'n_components': 5, 'n_iter': 7, 'random_state': 42},
        clustering_algorithm='kmeans',
        clustering_args={'n_clusters': 2, 'random_state': 0, 'n_init': "auto"})

    # run the pipeline:
    cc.fit(TEXTS)

    # show the results
    cc.show()

    # if results are unsatisfactory, refit with new selections
    cc.fit(
        projection_algorithm='pca', 
        projection_args={'n_components': 3},
        clustering_algorithm='hdbscan',
        clustering_args={'min_cluster_size': 10})

    cc.show()


    # still unsatisfied? you can keep projections, but change clustering params
    cc.fit(clustering_args={'min_cluster_size': 25})

    cc.show()

    # save when done
    cc.save("temp/cc_100k")

def doc_class_example():
    # Example usage
    my_clusterer = ClusterClassifier(
        batch_size=16,
        embed_model_name="all-MiniLM-L6-v2",
        clustering_algorithm="kmeans",
        clustering_args={'n_clusters': 5},
    )

    # Fit the model to the texts and get the embeddings, labels, and summaries
    embeddings, labels, summaries = my_clusterer.fit(TEXTS)

    # Visualize the clustering results
    my_clusterer.show()

    # Save the model
    my_clusterer.save("temp/cluster_classifier_5_clusters")

def doc_infer():
    inferred_labels, embeddings = cc.infer(TEXTS, top_k=3)

def doc_save():
    cc.save('temp/model_data')

def doc_load():
    cc.load('temp/model_data')

def doc_show_non_interactive():
    # Static plot (using Matplotlib)
    cc.show(interactive=False)

def doc_show_interactive():
    # Interactive plot (using Plotly)
    cc.show(interactive=True)

def test_func(func):
    global failures
    try:
        start = time.time()
        func()
        print(Fore.GREEN + f"{func} succeeded. Time taken: {time.time() - start:.2f}s")
    except Exception as e:
        print(Fore.RED + f"{func} failed with error: {e}")  # Print error to console
        failures += 1

def main():
    global failures
    start = time.time()

    test_func(readme_basic_fit)
    test_func(readme_basic_inference)
    test_func(readme_advanced)

    test_func(doc_class_example)
    test_func(doc_infer)
    test_func(doc_save)
    test_func(doc_load)
    test_func(doc_show_non_interactive)
    test_func(doc_show_interactive)

    if failures == 0:
        print(Fore.GREEN + f"All tests succeeded. Time taken: {time.time() - start:.2f}s")
    else:
        print(Fore.RED + f"{failures} tests failed.") 

main()