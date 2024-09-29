<h1>Flower classification</h1>

The model is trained to classify 16 flower classes. More about <a href="https://www.kaggle.com/datasets/l3llff/flowers">dataset</a>.
Training the model: <a href="https://www.kaggle.com/code/honeybadger128bit/flower-classification">this</a>.


Content:
<ul>
<li><a href="#flaskapi">Flask app API</a></li> 
<li><a href="#clfmodel">Classification model API</a></li>
<li><a href="#appdocs">App documentation</a></li>
</ul>

<div id="flaskapi">
<h3><u>Flask app API</u></h3>
<code>POST: /flowerclf/api/classification</code><br>
return: json like <code>{"probability": [list of probability], "labels": [list of labels]}</code><br>
in request budy must be field image
</div>

<h3><u>Classification model API</u></h3>

<div id="clfmodel">
<b>CLASS: app/model_proxy/ModelProxy(model: torch.nn.Module, data_preprocess: Callable[[np.array], torch.tensor])</b>
Wrapper class for Pytorch model.<br>

<b>Parameters:</b><br>
<b>model:</b> torch model;<br>
<b>data_preprocess:</b> any callable object for preprocessing input images, recommended using torch class <i>Compose</i>.<br>

<i>Class instance is callable</i>: <br>
receive RGB image like np.array <br>
return tuple like (probability, labels) where each element is np.array. <br>

For work model in [modelproxy_config.toml](app%2Fmodelproxy_config.toml) need write path to model weights and path to encoded labels.
Format for weights *.pth, for labels *.npy. <i>All files must be in app folder</i>.

<b>Example:</b><br>
<pre>
<code>
proxy = ModelProxy(**param)
prob, lab = proxy(image_in_numpy)
</code>
</pre>
</div>

<div id="appdocs">
<h3>App documentation</h3>

<h4>app</h4>

<b>app/stuff</b>: contain files with model weights and encoded labels.

<b>app/model_config.toml</b>: contain paths to model weights and encoded labels. <br>

<b>app/models/get_custom_densenet201(n_classes: int = 16) -> densenet201</b><br>
<b>Parameters:</b><br>
n_classes: int number of classes in model, by default 16. <br>
return: densenet201 with n_classes in classifier out <br>

<b>app/data_preprocessing_scripts.py</b>: contain torchvision.transforms.v2.Compose. <br>

<b>app/model_proxy.py</b>: Wrapper class for Pytorch model. <br>

<h4>Flask app</h4>

<b>flask_app/flask_app.py</b>: run from flask_app folder: <code>flask --app flask_app run</code> <br>

<h4>Streamlit app</h4>

<b>streamlit_app/home.py</b>: run from streamlit_app folder: <code>streamlit run home.py </code>
</div>
