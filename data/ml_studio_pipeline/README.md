        # ml_studio_pipeline

        Auto-generated Kedro project from ML Studio.
        Generated: 2026-05-25 18:28

        ## Quick start

        ```bash
        pip install -e ".[dev]"
        kedro run
        ```

        ## Pipeline steps


- Train Random Forest

        ## Notes

        - Scalers in this pipeline are fitted on the **full input dataset**.
          For production, split the data before scaling nodes and fit only on train.
        - The model is re-trained from scratch on each `kedro run`.
          Hyperparameters live in `conf/base/parameters.yml`.
        - Raw data is bundled in `data/01_raw/input.csv`.
          Replace it with new data and re-run to retrain on fresh data.
