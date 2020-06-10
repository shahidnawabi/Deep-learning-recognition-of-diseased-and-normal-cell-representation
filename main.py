from package import data_preprocessing as dp
from package import models
import warnings

base_dataset_path=r'..\Data_Set\resized_Images'
train_normal_images_path=r'\train_normal'
train_affected_images_path=r'\train_affected'
eval_normal_images_path=r'\test_normal'
eval_affected_images_path=r'\test_affected'


if __name__== "__main__":
	warnings.simplefilter('ignore', DeprecationWarning)
	train_features,train_labels,eval_features,eval_labels=dp.get_train_eval_data(base_dataset_path,train_normal_images_path,train_affected_images_path,eval_normal_images_path,eval_affected_images_path,0)
	print("Total Images for training :%d"%(len(train_features)))
	print("Total Images for evaluation :%d"%(len(eval_labels)))
	print("\n")
	models.logistic_regression(train_features,train_labels,eval_features,eval_labels)
	print("\n")
	models.support_vector_machine(train_features,train_labels,eval_features,eval_labels)
	print("\n")
	models.ensemble_logreg_svm(train_features,train_labels,eval_features,eval_labels)
	print("\n")
	models.resnet(50,True) #if training required then set skip = False , else True
	print("\n")
	models.resnet(152,True) #if training required then set skip = False , else True
	