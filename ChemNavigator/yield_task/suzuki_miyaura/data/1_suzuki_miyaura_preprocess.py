import pandas as pd
from pathlib import Path
from tqdm import tqdm

class SuzukiMiyauraProcessor:
    def __init__(self, root_dir=None):
        pass

    @property
    def raw_file_paths(self):
        # 假设原始文件在此路径
        return ['aap9112_Data_File_S1.xlsx']

    @property
    def processed_file_path(self):
        # 定义处理后文件的保存路径
        return 'suzuki_miyaura_smiles.csv'

    def process(self):
        """
        读取原始Excel文件，提取SMILES字符串，并保存为CSV文件。
        """

        print("Processing raw data file...")
        # load data
        df = pd.read_excel(self.raw_file_paths[0]).fillna('None')

        # --- SMILES 字符串字典 ---
        reactant_1_smiles = {'6-chloroquinoline': 'C1=C(Cl)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC',
                             '6-Bromoquinoline': 'C1=C(Br)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC',
                             '6-triflatequinoline': 'C1C2C(=NC=CC=2)C=CC=1OS(C(F)(F)F)(=O)=O.CCC1=CC(=CC=C1)CC',
                             '6-Iodoquinoline': 'C1=C(I)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC',
                             '6-quinoline-boronic acid hydrochloride': 'C1C(B(O)O)=CC=C2N=CC=CC=12.Cl.O',
                             'Potassium quinoline-6-trifluoroborate': '[B-](C1=CC2=C(C=C1)N=CC=C2)(F)(F)F.[K+].O',
                             '6-Quinolineboronic acid pinacol ester': 'B1(OC(C(O1)(C)C)(C)C)C2=CC3=C(C=C2)N=CC=C3.O'}
        reactant_2_smiles = {'2a, Boronic Acid': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B(O)O',
                             '2b, Boronic Ester': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B4OC(C)(C)C(C)(C)O4',
                             '2c, Trifluoroborate': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1[B-](F)(F)F.[K+]',
                             '2d, Bromide': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1Br'}
        catalyst_smiles = {'Pd(OAc)2': 'CC(=O)O~CC(=O)O~[Pd]'}
        ligand_smiles = {'P(tBu)3': 'CC(C)(C)P(C(C)(C)C)C(C)(C)C', 'P(Ph)3 ': 'c3c(P(c1ccccc1)c2ccccc2)cccc3',
                         'AmPhos': 'CC(C)(C)P(C1=CC=C(C=C1)N(C)C)C(C)(C)C', 'P(Cy)3': 'C1(CCCCC1)P(C2CCCCC2)C3CCCCC3',
                         'P(o-Tol)3': 'CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C',
                         'CataCXium A': 'CCCCP(C12CC3CC(C1)CC(C3)C2)C45CC6CC(C4)CC(C6)C5',
                         'SPhos': 'COc1cccc(c1c2ccccc2P(C3CCCCC3)C4CCCCC4)OC',
                         'dtbpf': 'CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.[Fe]',
                         'XPhos': 'P(c2ccccc2c1c(cc(cc1C(C)C)C(C)C)C(C)C)(C3CCCCC3)C4CCCCC4',
                         'dppf': 'C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.[Fe+2]',
                         'Xantphos': 'O6c1c(cccc1P(c2ccccc2)c3ccccc3)C(c7cccc(P(c4ccccc4)c5ccccc5)c67)(C)C', 'None': ''}
        reagent_1_smiles = {'NaOH': '[OH-].[Na+]', 'NaHCO3': '[Na+].OC([O-])=O', 'CsF': '[F-].[Cs+]',
                            'K3PO4': '[K+].[K+].[K+].[O-]P([O-])([O-])=O', 'KOH': '[K+].[OH-]',
                            'LiOtBu': '[Li+].[O-]C(C)(C)C', 'Et3N': 'CCN(CC)CC', 'None': ''}
        solvent_1_smiles = {'MeCN': 'CC#N.O', 'THF': 'C1CCOC1.O', 'DMF': 'CN(C)C=O.O', 'MeOH': 'CO.O',
                            'MeOH/H2O_V2 9:1': 'CO.O', 'THF_V2': 'C1CCOC1.O'}

        reactions_data = []
        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing reactions"):
            ligand = ligand_smiles[row['Ligand_Short_Hand']]
            catalyst_complex = catalyst_smiles[row['Catalyst_1_Short_Hand']]
            if ligand:
                catalyst_complex += '.' + ligand

            reagent = reagent_1_smiles[row['Reagent_1_Short_Hand']]
            solvent = solvent_1_smiles[row['Solvent_1_Short_Hand']]

            reactions_data.append({
                'reactant1_smiles': reactant_1_smiles[row['Reactant_1_Name']],
                'reactant2_smiles': reactant_2_smiles[row['Reactant_2_Name']],
                'catalyst': catalyst_complex,
                'reagent': reagent,
                'solvent':solvent,
                'product_smiles': 'C1=C(C2=C(C)C=CC3N(C4OCCCC4)N=CC2=3)C=CC2=NC=CC=C12',
                'y_val': row['Product_Yield_PCT_Area_UV']
            })

        output_df = pd.DataFrame(reactions_data)
        output_df.to_csv(self.processed_file_path, index=False)
        print(f"Data successfully processed and saved to {self.processed_file_path}")


if __name__ == '__main__':
    # 创建一个处理器实例
    processor = SuzukiMiyauraProcessor()
    # 执行处理流程
    processor.process()