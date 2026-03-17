# Week 4: Multi Camera Tracking

## Evaluation

* He cogido la implemetación que venía en el dataset que se usó para el AI CITY challenge. **No usa TrackEval**.

* El script estaba pensado para hacer evaluación de todas las secuencias a la vez (un txt por gt, y un txt por preds). Lo he cambiado y he generado un `gt_S0X.txt` por cada secuencia que estan guardados en `data/gt/`. Los he generado con el script de `src/utils/split_gt_by_sequence.py`. 

* El script necesitaba versiones de librerías bastante especifícas, asi que he tenido que hacer otro environment para evaluar para no romper ninguna dependencia con vosotros. Se llama `environment_eval.yml`.

* Para evaluar hay que hacer: 
```bash
python -m src.main evaluate --gt data/gt/gt_S03.txt --pred data/preds/preds_S03.txt --seq "S03"
```
en cuanto tengamos los preds listos. De momento, lo he probado con gt contra gt para la secuencia S03 y da IDF1 100.0 asi que bien :). También hay un argumento de `--roidir` que no hace falta cambiar porque para todas las secuencias es el mismo path. Lo usan para filtrar detecciones fuera de la zona válida antes de calcular métricas.