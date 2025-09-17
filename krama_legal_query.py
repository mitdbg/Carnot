import os

import chromadb
import pandas as pd

import carnot
from carnot.constants import Model
from carnot.core.elements.records import DataRecordCollection
from carnot.core.lib.schemas import TextFile


class LegalTextFileDataset(carnot.IterDataset):
    """
    TextFileDataset returns a dictionary for each text file in a directory. Each dictionary contains the
    filename and contents of a single text file in the directory.
    """
    def __init__(self, path: str, **kwargs) -> None:
        """
        Constructor for the `BaseFileDataset` class.

        Args:
            path (str): The path to the file
            kwargs (dict): Keyword arguments containing the `Dataset's` id and file-specific `Schema`
        """
        # check that path is a valid file or directory
        assert os.path.isfile(path) or os.path.isdir(path), f"Path {path} is not a file nor a directory"

        # get list of filepaths
        self.filepaths = []
        if os.path.isfile(path):
            self.filepaths = [path]
        else:
            self.filepaths = []
            for root, _, files in os.walk(path):
                for file in files:
                    fp = os.path.join(root, file)
                    self.filepaths.append(fp)
            self.filepaths = sorted(self.filepaths)

        # call parent constructor to set id, operator, and schema
        super().__init__(**kwargs)

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a dictionary with the filename and contents of the text file at the specified `idx`.

        Args:
            idx (int): The index of the item to return

        Returns:
            dict: A dictionary with the filename and contents of the text file.

            .. code-block:: python

                {
                    "filename": "file.txt",
                    "contents": "text content here",
                }
        """
        filepath = self.filepaths[idx]
        contents = None
        if filepath.endswith(".csv"):
            contents = pd.read_csv(filepath, encoding="ISO-8859-1").to_string(index=False)

        else:
            with open(filepath, encoding='utf-8') as file:
                contents = file.read()

        return {"filename": os.path.basename(filepath), "contents": contents}

# def run_carnot_unopt():
#     # "Does there exist a metropolitan area in which the number of reports of identity theft exceeded the number of reports of fraud in 2024? Answer with Yes or No. No explanation needed."
#     ds1 = LegalTextFileDataset(id="legal-dataset-1", schema=TextFile, path="../Kramabench/data/legal/input/")
#     ds1 = ds1.sem_filter("has metropolitan area stats on identity thefts")
#     ds1 = ds1.sem_add_columns(cols=[{
#         "name": "metro_area_to_identity_thefts",
#         "type": dict,
#         "desc": "dictionary mapping metro areas to the number of identity thefts."
#     }])

#     config = carnot.QueryProcessorConfig(
#         execution_strategy="parallel",
#         available_models=[Model.GPT_4o],
#         max_workers=20,
#     )
#     return ds1.run(config=config)


def run_carnot_unopt():
    ds1 = LegalTextFileDataset(id="legal-dataset-1", schema=TextFile, path="../Kramabench/data/legal/input/")
    ds1 = ds1.sem_filter("the file contains the number of identity theft reports in 2024")

    ds2 = LegalTextFileDataset(id="legal-dataset-2", schema=TextFile, path="../Kramabench/data/legal/input/")
    ds2 = ds2.sem_filter("the file contains the number of identity theft reports in 2001")

    ds3 = ds1.sem_join(ds2, condition="The join condition is TRUE; this is an outer join so return TRUE for all records.")
    ds3 = ds3.sem_add_columns(cols=[{"name": "ratio", "type": float, "desc": "the number of identity theft reports in 2024 divided by the number of identity theft reports in 2001"}])

    config = carnot.QueryProcessorConfig(
        execution_strategy="parallel",
        available_models=[Model.GPT_4o],
        max_workers=20,
    )
    return ds3.run(config=config)

def run_carnot_compute(question):
    ds = carnot.TextFileContext(
        "../Kramabench/data/legal/input/",
        id="legal-dataset",
        description="Files containing statistics from the FTC on identity theft, fraud, and other reports.",
    )
    ds = ds.compute(
        question,
    )

    config = carnot.QueryProcessorConfig(available_models=[Model.GPT_4o], progress=False)
    return ds.run(config=config, progress=False)

def clear_cache():
    # remove context files
    context_dir = os.path.join(carnot.constants.PZ_DIR, "contexts")
    for filename in os.listdir(context_dir):
        os.remove(os.path.join(context_dir, filename))

    # clear collection
    chroma_dir = os.path.join(carnot.constants.PZ_DIR, "chroma")
    chroma_client = chromadb.PersistentClient(chroma_dir)
    try:  # noqa: SIM105
        chroma_client.delete_collection("contexts")
    except chromadb.errors.NotFoundError:
        # collection does not exist, no need to delete
        pass

def save_results(results: DataRecordCollection, id):
    os.makedirs("cidr-legal-results/", exist_ok=True)

    # write the execution results to disk
    results.execution_stats.to_json(f"cidr-legal-results/{id}_stats.json")
    results_df = results.to_df()
    results_df.to_csv(f"cidr-legal-results/{id}.csv", index=False)

    # return the final answer
    # answer = results_df.iloc[0]["final_answer"]
    for k, v in results_df.iloc[0].to_dict().items():
        if k.startswith("result-"):
            answer = v
            print(f"ANSWER IS: {answer}")
            break


if __name__ == "__main__":
    clear_cache()
    # carnot_unopt_out = run_carnot_unopt()
    # save_results(carnot_unopt_out, "carnot-unopt-1")

    carnot_compute_out = run_carnot_compute("Give the ratio of identity theft reports in 2024 vs 2001? Round to 4 decimal places")
    save_results(carnot_compute_out, "carnot-compute-1")
