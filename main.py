import gradio as gr 
from load_model import PrePro

# Create an instance of PrePro class
prs = PrePro()

# Define the function for the image classifier
def image_classifier(inp):
    # Preprocess the input image
    img = prs.preprocess(inp)
    
    # Get the prediction from the model
    out = prs.predict(img)
    
    # Get the probabilities and labels for the top predictions
    results = prs.probability(out)
    
    # Return the results
    return results

# Create a Gradio interface with the image classifier function
demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label")

# Launch the Gradio interface
demo.launch()


