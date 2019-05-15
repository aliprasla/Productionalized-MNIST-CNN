const train_model_function = function(model_training_text){
	$.post("/train",null,
	   function(data,status){
		if (status == "success"){
			alert("Model training successful");
			model_training_text.text("");
		}
		else{
			model_training_text.text("Model Training Failed. Data: "+data);
			console.log(status);
		}
	});
	
	console.log("Post Request Complete");
	model_training_text.text("Model is currently training");

	return model_training_text;
}
