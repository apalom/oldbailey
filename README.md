# oldbailey
 MLprojectFall2020

# Avg Perceptron Oct 21 0:21
Average perceptron on bag-of-words training data augmented by one-hot encoded metadata features (age, gender, offence cat).
5-fold cv; Learning rate = 1; 5 epochs.
Peak training validation accuracy ~0.88.

# Meta features
['defendant_age', 'defendant_gender', 'num_victims', 'victim_genders', 'offence_category', 'offence_subcategory']

# Offence category values
['kill', 'violentTheft', 'damage', 'sexual', 'theft', 'deception', 'royalOffences', 'miscellaneous', 'breakingPeace']
