from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# build a model with base model being MobileNetV2
# Final model:
# 	MobileNetV2 => AVGPOOL => FLATTEN => DENSE => DROP => DENSE
class Detector:
	@staticmethod

	def build(inputShape):
		# initialize the base model
		baseModel = MobileNetV2(weights="imagenet", 
			include_top=False,
			input_tensor=Input(shape=inputShape))

		# build additional layer on top of the baseModel
		headModel = baseModel.output
		headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
		headModel = Flatten(name="flatten")(headModel)
		headModel = Dense(128, activation="relu")(headModel)
		headModel = Dropout(0.5)(headModel)
		headModel = Dense(2, activation="softmax")(headModel)

		# build the final model
		model = Model(inputs=baseModel.input, outputs=headModel)

		# freeze the weights of the baseModel
		for layer in baseModel.layers:
			layer.trainable = False 

		return model
