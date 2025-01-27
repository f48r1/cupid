QUERY_TEMPLATE =\
"""with cav_att as
(select cs.molregno,cast(standard_value as real), cs.canonical_smiles, md.pref_name, md.chembl_id, a.tid, a.assay_id, organism from activities ac 
join assays a on ac.assay_id=a.assay_id join compound_structures cs on ac.molregno=cs.molregno
join target_dictionary td on td.tid=a.tid
join molecule_dictionary md on ac.molregno=md.molregno
where a.tid in ({tid_list}) and ac.standard_type='IC50' and (standard_relation='='or ( standard_relation='>=' and standard_value > 10000) or
(standard_relation='>' and standard_value>10000) or (standard_relation='<' and standard_value<10000) or
(standard_relation='<=' and standard_value<10000))),
valori as (select array_agg(standard_value) va, organism, molregno,tid,canonical_smiles, pref_name, chembl_id from cav_att group by molregno,tid,organism,canonical_smiles, pref_name, chembl_id)
select * from valori order by array_length(va,1);"""

# FIXME Put your address SQL for chembl database
ADDRESS = "..."

if __name__ == '__main__':
    import argparse
    import os
    import pandas as pd
    from src.paths import DIR_RAWDATASET
    from src.target_configs import TIDS

    parser = argparse.ArgumentParser(description='Parameters to initialize raw dataset')

    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset')

    parser.add_argument('--dir_rawdataset', type=str, required=False, default=DIR_RAWDATASET,
                        help='directory for raw dataset files')
    
    parser.add_argument('--address', type=str, required=False, default=ADDRESS,
                        help='directory for raw dataset files')
    
    args,unk = parser.parse_known_args()
    if unk:
        print("Unknown arguments passed:", ", ".join(unk))

    os.makedirs(
        os.path.join(args.dir_rawdataset), 
        exist_ok=True
        )
    
    query_df = pd.read_sql_query(QUERY_TEMPLATE.format(tid_list=TIDS[args.dataset]), args.address)
    query_df.to_csv(os.path.join(args.dir_rawdataset,args.dataset) + '.csv', index=False)