# Cian duplicates task

## Run

```
docker build -t cian_duplicates_zhukov .
docker run -p 80:80 cian_duplicates_zhukov
```

### Что сделано:

* Небольшой эксплорейшен `explore.ipynb`
* обучен w2v на текстах description `w2v_learning.ipynb`
* построены различные фичи на текстах, гео-координатах и описании, на прочих количественных показателях `feature_engineering_and_model.ipynb`
* тюнинг xgboost, результат precision = 0.82, recall = 0.75 `feature_engineering_and_model.ipynb`

### Что можно было бы сделать еще:

* отдельные модели для аренды и продажи (чтобы использовать специфические фичи)
* другие модели для сравнения текстов (fasttext, doc2vec другими способами)
* прочие простые признаки там, где не очень большой процент nan-ов
