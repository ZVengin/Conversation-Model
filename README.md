# Conversation-Model
This project aims to implement a End-to-End neural network based conversation model baseline. This baseline model consists of two components, one is RNN encoder, the other is RNN decoder. In this project, regarding RNN type, GRU is adopted in both encoder and decoder. 
## Get started
### prerequisizte
If you want to run this project without any problem, it is better create a new enviornment for this project. Regarding the environment of this project, `pytorch 0.4`, `tensorflow`, `python 3.6` and `tensorboard_logger` are used. With respect to `tensorboard_logger`, it provides a convenient way to use write log to a log file, however, this reposistory relies on tensorflow repository. Therefore, you should install tensorflow cpu version first at least. Then you can follow [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger) to install it. In order to make whole process easier, `requirement.yml` file is inclueded in this repository. Therefore, you can run following command to install required environment.
```
conda-env create -n conversation_model -f requirement.yml
```

### Prepare data
In this project, [Cornell Movie-Dialogue Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) is used. Because the data is text format, you need to preprocess conversation datat before training model. Even though `Cornell Movie Corpus` is used here, I think this dataset is not suitable for training a conversation model since there is too much noise in this dataset and it is not large enough to train a good conversation model.
#### Generate conversation pair and build vocabulary
```
python generate_conversation.py
```

## Train model
Default setting is given in `generate_configure.py` script, if you want to make some modifications to these parameters, you only need to modify corresponding parameters in this script. After that, run following command to start to train model.
```
python run.py --mode train --exp_dir place/of/storing/experiment/data --gen_config True
```

## Test model
Similar to training command, only `mode` needs to be modified as `test`.
```
python run.py --mode test --exp_dir place/of/storing/experiment_data 
```
## Authors

* Zvengin

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
