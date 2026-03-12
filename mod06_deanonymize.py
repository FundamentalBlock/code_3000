import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    keys = ["age", "zip3", "gender"]
    merged = anon_df.merge(aux_df, on=keys, how="inner")

    # Count how many auxiliary names each anon record matched
    match_counts = merged.groupby("anon_id")["name"].count()
    unique_anon_ids = match_counts[match_counts == 1].index

    # Count how many anon records each auxiliary name matched
    name_counts = merged.groupby("name")["anon_id"].count()
    unique_names = name_counts[name_counts == 1].index

    unique_matches = merged[
        merged["anon_id"].isin(unique_anon_ids) & merged["name"].isin(unique_names)
    ][["anon_id", "name"]].rename(columns={"name": "matched_name"})

    return unique_matches.reset_index(drop=True)


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return len(matches_df) / len(anon_df)
