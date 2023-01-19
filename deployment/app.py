from fastai.vision.all import load_learner
import gradio as gr


car_labels = [
   'Bus',
   'Convertible',
   'Hatchback', 
   'Limousine', 
   'Micro', 
   'Minivan', 
   'Muscle car', 
   'Sedan', 
   'Sports car', 
   'SUV', 
   'Truck'
]

model = load_learner('models/car-recognizer-v0.pkl')


def recognize_image(image):
  pred, idx, probs = model.predict(image)
  return dict(zip(car_labels, map(float, probs)))

def greet(name):
    return "Hello " + name + "!!"


image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = [
    'test_images/00.jpg',
    'test_images/bus.jpg',
    'test_images/minivan.jpg',
    'test_images/musclecar.jpg',
    'test_images/sedan.jpg',
    'test_images/sports.jpg',
    'test_images/suv.jpg',
    'test_images/truck.jpg'
]

iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(share=True)