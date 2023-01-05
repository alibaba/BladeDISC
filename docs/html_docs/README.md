# To generate documentation htmls:

* Install the prerequisites:

```shell
conda install -c conda-forge pandoc
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
```

* Refer to [workaround](workaround_passpipeline_html.sh) if the notebook
  documents with run results are to be demo on websites.

* Generate htmls with `make html`.

* To check the generated htmls:

```shell
cd build/html
python -m http.server 8080
```

* Deploy to Github Page

```shell
# in build/html path
git init
git checkout --orphan main_docs
touch .nojekyll
git add *
git add .nojekyll
git commit -a -m "Documentation verX.X.X"
git push origin --force main_docs
```

TODO: Move the flow to ReadtheDocs if it's not that necessary to demo notebook
docs with run results. Revisit this in future.
