import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions
import numpy as np
import argparse

def canonicalize_with_dict(smi, can_smi_dict={}):
    if smi not in can_smi_dict.keys():
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    else:
        return can_smi_dict[smi]
   
#####For intermediate generation
def generate_bh_mechanism(df):
    df = df.copy()

    #Ligand binding and pre-catalyst formation
    temp1 = '[*:2][P:1]([*:3])([*:4]).[Pd:5]>>[*:2][P:1]([*:3])([*:4])->[Pd:5]'
    Pd = Chem.MolFromSmiles('O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F')
    pd_catalyst = Chem.MolToSmiles(Chem.MolFromSmiles('O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F'))
    rxn1 = rdChemReactions.ReactionFromSmarts(temp1)
    cat = []

    #Active catalyst formation
    temp2 = '[P:4]->[Pd:5]5(OS(=O)(=O)C(F)(F)F)c6ccccc6-c6ccccc6N~5>>[P:4]->[Pd:5]'
    rxn2 = rdChemReactions.ReactionFromSmarts(temp2)
    act = []

    #Oxidative addition i.e. substrate binding
    temp3 = '[F,Cl,Br,I:10]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3].[Pd:5]>>[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[Pd:5]-[F,Cl,Br,I:10]'
    rxn3 = rdChemReactions.ReactionFromSmarts(temp3)
    int_a = []

    #Amination
    temp4 = '[N:1].[Pd:5]>>[N:1]->[Pd:5]'
    methylaniline = 'Cc1ccc(N)cc1'
    methylaniline_mol = Chem.MolFromSmiles(methylaniline)
    rxn4 = rdChemReactions.ReactionFromSmarts(temp4)
    int_b = []

    #Proton abstraction
    temp5 = '[N:8]=[*:9].[c:1][N:2]->[Pd:5][F,Cl,Br,I:10]>>[N+:8]=[*:9].[c:1][N:2]-[Pd:5].[F,Cl,Br,I-:10]'
    rxn5 = rdChemReactions.ReactionFromSmarts(temp5)
    conj_a = []
    int_c = []

    #Reductive elimination
    temp6 = '[N:1][Pd:2][c:3]>>[N:1]-[c:3]'
    rxn6 = rdChemReactions.ReactionFromSmarts(temp6)
    pdt = []

    #a = []
    for i, row in df.iterrows():
        #print('Reaction no.:', i)

        #Ligand binding and pre-catalyst formation
        reacts1 = (Chem.MolFromSmiles(row['Ligand']), Pd)
        catalysts = rxn1.RunReactants(reacts1)
        cat_smiles = set([Chem.MolToSmiles(mol[0]) for mol in catalysts])
        #assert len(cat_smiles) == 1
        cat.append(list(cat_smiles)[0])
        #print("No. of possible pre-catalysts: ", len(list(cat_smiles)))

        #Active catalyst formation
        reacts2 = (Chem.MolFromSmiles(list(cat_smiles)[0]),)
        active_catalysts = rxn2.RunReactants(reacts2)
        act_smiles = set([Chem.MolToSmiles(mol[0]) for mol in active_catalysts])
        #assert len(act_smiles) == 1
        act.append(list(act_smiles)[0])
        #print("No. of possible active catalysts: ", len(list(act_smiles)))

        #Oxidative addition i.e. substrate binding
        reacts3 = (Chem.MolFromSmiles(row['Aryl halide']),Chem.MolFromSmiles(list(act_smiles)[0]))
        oxads = rxn3.RunReactants(reacts3)
        int_a_smiles = set([Chem.MolToSmiles(mol[0]) for mol in oxads])
        #assert len(int_a_smiles) == 1
        int_a.append(list(int_a_smiles)[0])
        #print("No. of possible intermediate-I: ", len(list(int_a_smiles)))
        
        #Amination
        reacts4 = (methylaniline_mol,Chem.MolFromSmiles(list(int_a_smiles)[0]))
        binds = rxn4.RunReactants(reacts4)
        int_b_smiles = set([Chem.MolToSmiles(mol[0]) for mol in binds])
        #assert len(int_b_smiles) == 1
        int_b.append(list(int_b_smiles)[0])
        #print("No. of possible intermediate-II: ", len(list(int_b_smiles)))

        #Proton abstraction
        reacts5 = (Chem.MolFromSmiles(row['Base']),Chem.MolFromSmiles(list(int_b_smiles)[0]))
        abstracts = rxn5.RunReactants(reacts5)
        conj_a_smiles = set([Chem.MolToSmiles(mol[0]) for mol in abstracts])
        int_c_smiles = set([Chem.MolToSmiles(mol[1]) for mol in abstracts])
        #assert len(conj_a_smiles) == 1
        #assert len(int_c_smiles) == 1
        if len(list(conj_a_smiles))>1:
            #print(list(conj_a_smiles)[1])
            conj_a.append(list(conj_a_smiles)[1])
        else:
            conj_a.append(list(conj_a_smiles)[0])
        int_c.append(list(int_c_smiles)[0])
        #print("No. of possible intermediate-III: ", len(list(int_c_smiles)))
        #print("No. of possible conjugated acid: ", len(list(conj_a_smiles)))

        #Reductive elimination
        reacts6 = (Chem.MolFromSmiles(list(int_c_smiles)[0]),)
        redels = rxn6.RunReactants(reacts6)
        pdt_smiles = set([Chem.MolToSmiles(mol[0]) for mol in redels])
        #assert len(pdt_smiles) == 1
        pdt.append(list(pdt_smiles)[0])
        #print("No. of possible products: ", len(list(pdt_smiles)))

        #multiple possibilities
        '''
        if len(list(conj_a_smiles)) > 1:
            a.append(list(conj_a_smiles)[0])
            #print(f'Reaction {i} has {len(list(conj_a_smiles))} possibilities')
            #print({list(conj_a_smiles)})
        '''
    df['pre-catalyst'] = cat 
    df['active_catalyst'] = act
    df['intermediate_I'] = int_a
    df['intermediate_II'] = int_b
    df['conjugated_acid'] = conj_a
    df['intermediate_III'] = int_c
    df['product'] = pdt

    '''
    mechs = []
    can_smiles_dict = {}
    for i, row in df.iterrows():
        aryl_halide = canonicalize_with_dict(row['Aryl halide'], can_smiles_dict)
        can_smiles_dict[row['Aryl halide']] = aryl_halide
        ligand = canonicalize_with_dict(row['Ligand'], can_smiles_dict)
        can_smiles_dict[row['Ligand']] = ligand
        base = canonicalize_with_dict(row['Base'], can_smiles_dict)
        can_smiles_dict[row['Base']] = base
        additive = canonicalize_with_dict(row['Additive'], can_smiles_dict)
        can_smiles_dict[row['Additive']] = additive
        reactants = f"{ligand}.{pd_catalyst}.{row['cat']}.{aryl_halide}.{row['act']}.{methylaniline}.{row['int_a']}.{base}.{row['int_b']}.{row['int_c']}"
        products = f"{row['cat']}.{row['act']}.{row['int_a']}.{row['int_b']}.{row['conj_a']}.{row['int_c']}.{row['pdt']}"
        #reactants = f"{aryl_halide}.{methylaniline}.{pd_catalyst}.{ligand}.{base}.{additive}"
        mechs.append(f"{reactants}.{additive}>>{products}")
    '''
    #print(len(a))
    return df   #mech
    
def generate_bh_total(df):

    df = df.copy()
    mechs = []
    methylaniline = 'Cc1ccc(N)cc1'
    methylaniline_mol = Chem.MolFromSmiles(methylaniline)
    pd_catalyst = Chem.MolToSmiles(Chem.MolFromSmiles('O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F'))
    can_smiles_dict = {}
    for i, row in df.iterrows():
        aryl_halide = canonicalize_with_dict(row['Aryl halide'], can_smiles_dict)
        can_smiles_dict[row['Aryl halide']] = aryl_halide
        ligand = canonicalize_with_dict(row['Ligand'], can_smiles_dict)
        can_smiles_dict[row['Ligand']] = ligand
        base = canonicalize_with_dict(row['Base'], can_smiles_dict)
        can_smiles_dict[row['Base']] = base
        additive = canonicalize_with_dict(row['Additive'], can_smiles_dict)
        can_smiles_dict[row['Additive']] = additive
        #reactants = f"{row['intermediate_I']}.{methylaniline}.{base}.{additive}"
        products = f"{row['product']}"
        reactants = f"{aryl_halide}.{methylaniline}.{pd_catalyst}.{ligand}.{base}.{additive}"
        intermediates = f"{row['intermediate_I']}.{row['intermediate_II']}.{row['conjugate_acid']}.{row['intermediate_III']}"
        mechs.append(f"{reactants}>>{intermediates}>>{products}")
    return mechs

def generate_df_mechanism(df):
    df = df.copy()

    #Proton abstraction
    fwd_template = '[C:1][OH:2].[N:3]=[*:4]>>[C:1][O-:2].[NH+:3]=[*:4]'
    rxn1 = rdChemReactions.ReactionFromSmarts(fwd_template)
    oxide = []
    conj_a = []

    #Intermediate formation
    fwd_template = '[C:1][O-:2].[S:3][F:4]>>[C:1][O+0:2][S:3]'
    rxn2 = rdChemReactions.ReactionFromSmarts(fwd_template)
    sulfoxide = []

    #Substitution
    fwd_template1 = '[C@H:1][O:2][S:3]>>[C@@H:1][F:4]'
    fwd_template2 = '[C:1][O:2][S:3]>>[C:1][F:4]'
    rxn3 = rdChemReactions.ReactionFromSmarts(fwd_template1)
    rxn4 = rdChemReactions.ReactionFromSmarts(fwd_template2)
    pdt = []

    for i, row in df.iterrows():
        #print('Reaction no.: ', i)
        #Proton abstraction
        reacts1 = (Chem.MolFromSmiles(row['alcohol']), Chem.MolFromSmiles(row['base']))
        abstracts = rxn1.RunReactants(reacts1)
        oxide_smiles = set([Chem.MolToSmiles(mol[0]) for mol in abstracts])
        if len(list(oxide_smiles)) > 1:
            oxide.append(list(oxide_smiles)[1])
        else:
            oxide.append(list(oxide_smiles)[0])
        conj_a_smiles = set([Chem.MolToSmiles(mol[1]) for mol in abstracts])
        conj_a.append(list(conj_a_smiles)[0])
        #print("No. of possible conjugated acid: ", len(list(conj_a_smiles)))
        #print("No. of possible oxide: ", len(list(oxide_smiles)))
       
        #Intermediate formation
        reacts2 = (Chem.MolFromSmiles(list(oxide_smiles)[0]), Chem.MolFromSmiles(row['sulfonyl_fluoride']))
        anchors = rxn2.RunReactants(reacts2)
        sulfoxide_smiles = set([Chem.MolToSmiles(mol[0]) for mol in anchors])
        #assert len(sulfoxide_smiles) == 1
        sulfoxide.append(list(sulfoxide_smiles)[0])
        #print("No. of possible sulfoxide: ", len(list(sulfoxide_smiles)))

        #Substitution
        try:
            reacts3 = (Chem.MolFromSmiles(list(sulfoxide_smiles)[0]),)
            attacks = rxn3.RunReactants(reacts3)
            pdt_smiles = set([Chem.MolToSmiles(mol[0]) for mol in attacks])
            #assert len(pdt_smiles) == 1
            pdt.append(list(pdt_smiles)[0])
            #print("No. of possible product: ", len(list(pdt_smiles)))
        except IndexError:
            reacts3 = (Chem.MolFromSmiles(list(sulfoxide_smiles)[0]),)
            attacks = rxn4.RunReactants(reacts3)
            pdt_smiles = set([Chem.MolToSmiles(mol[0]) for mol in attacks])
            #assert len(pdt_smiles) == 1
            pdt.append(list(pdt_smiles)[0])
            #print("No. of possible product: ", len(list(pdt_smiles)))

        ##multiple oxides
        #if len(list(oxide_smiles)) > 1:
            #print(f'Reaction {i} has oxide {len(list(oxide_smiles))}')
            #print(list(oxide_smiles))
    df['oxide'] = oxide
    df['conjugate_acid'] = conj_a
    df['sulfoxide'] = sulfoxide
    df['product'] = pdt
    F = Chem.MolToSmiles(Chem.MolFromSmiles('[F-]'))
    df['fluoride'] = F
    
    mechs = []
    can_smiles_dict = {}
    for i, row in df.iterrows():
        alcohol = canonicalize_with_dict(row['alcohol'], can_smiles_dict)
        can_smiles_dict[row['alcohol']] = alcohol
        sulfonyl_fluoride = canonicalize_with_dict(row['sulfonyl_fluoride'], can_smiles_dict)
        can_smiles_dict[row['sulfonyl_fluoride']] = sulfonyl_fluoride
        base = canonicalize_with_dict(row['base'], can_smiles_dict)
        can_smiles_dict[row['base']] = base

        intermediates = f"{row['oxide']}.{row['conjugate_acid']}.{row['sulfoxide']}"
        #intermediates = f"{row['oxide']}.{row['conjugate_acid']}>>{row['sulfoxide']}"
        products = f"{row['product']}"
        reactants = f"{alcohol}.{base}.{sulfonyl_fluoride}"
        mechs.append(f"{reactants}>>{intermediates}>>{products}")
    
    #df['mech'] = mechs
    return mechs

def generate_ns_total(df):
    df = df.copy()
    mechs = []
    can_smiles_dict = {}
    for i, row in df.iterrows():
        cpa = canonicalize_with_dict(row['CPA'], can_smiles_dict)
        can_smiles_dict[row['CPA']] = cpa
        substrate = canonicalize_with_dict(row['Substrate'], can_smiles_dict)
        can_smiles_dict[row['Substrate']] = substrate
        thiol = canonicalize_with_dict(row['Thiol'], can_smiles_dict)
        can_smiles_dict[row['Thiol']] = thiol
        products = f"{row['product']}"
        reactants = f"{cpa}.{substrate}.{thiol}"
        #intermediates = f"{row['active_substrate']}>>{row['pre_product']}"
        intermediates = f"{row['active_substrate']}.{row['pre_product']}"
        mechs.append(f"{reactants}>>{intermediates}>>{products}")
    return mechs
       
def generate_sc_total(df):

    df = df.copy()
    mechs = []
    can_smiles_dict = {}
    for i, row in df.iterrows():
        Sub1 = canonicalize_with_dict(row['Sub1'], can_smiles_dict)
        can_smiles_dict[row['Sub1']] = Sub1
        Sub2 = canonicalize_with_dict(row['Sub2'], can_smiles_dict)
        can_smiles_dict[row['Sub2']] = Sub2
        base = canonicalize_with_dict(row['Base'], can_smiles_dict)
        can_smiles_dict[row['Base']] = base
        Solvent = canonicalize_with_dict(row['Solvent'], can_smiles_dict)
        can_smiles_dict[row['Solvent']] = Solvent
        Product = canonicalize_with_dict(row['Product'], can_smiles_dict)
        can_smiles_dict[row['Product']] = Product
        #reactants = f"{row['intermediate_I']}.{methylaniline}.{base}.{additive}"
        products = f"{row['Product']}"
        reactants = f"{row['active-catalyst-smiles-dative']}.{Sub2}.{Sub1}.{base}.{Solvent}"
        intermediates = f"{row['new_smiles-dative']}.{row[ 'TM-smiles-dative']}"
        mechs.append(f"{reactants}>>{intermediates}>>{products}")
    return mechs
    
def generate_cm_total(df):

    df = df.copy()
    mechs = []
    can_smiles_dict = {}
    for i, row in df.iterrows():
        olefin1 = canonicalize_with_dict(row['Substrate1'], can_smiles_dict)
        can_smiles_dict[row['Substrate1']] = olefin1
        olefin2 = canonicalize_with_dict(row['Substrate2'], can_smiles_dict)
        can_smiles_dict[row['Substrate2']] = olefin2
        catalyst = canonicalize_with_dict(row['Catalyst'], can_smiles_dict)
        can_smiles_dict[row['Catalyst']] = catalyst
        #reactants = f"{row['intermediate_I']}.{methylaniline}.{base}.{additive}"
        products = f"{row['Product']}"
        reactants = f"{catalyst}.{olefin1}.{olefin2}"
        intermediates = f"{row['Active_catalyst(R1)']}.{row['Intermediate1(I1)']}.{row['Intermediate2(I2)']}.{row['Intermediate3(I3)']}"
        mechs.append(f"{reactants}>>{intermediates}>>{products}")
    return mechs

def scrambler(df):
    df = df.copy()
    nrows = round(0.7*len(df.loc[:, 'Yield']))
    df.loc[range(nrows),'Yield'] = np.random.permutation(df.loc[range(nrows), 'Yield'])
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('sheet', default=0)
    args = parser.parse_args()
    sheet_name = args.sheet
    #df = pd.read_csv(f'Dataset/bh_{sheet_name}.csv')   
    df = pd.read_excel('Dataset/SC/sc_data.xlsx', sheet_name='FullCV_'+str(sheet_name))
    #df = scrambler(df)  #Run only for scrambling
    rxns = generate_sc_total(df)
    df['concat_smiles'] = rxns
    smi = df['concat_smiles'].to_numpy().reshape((-1, 1))
    yld = df['Yield'].to_numpy().reshape((-1,1))
    final = np.concatenate((smi, yld), axis=1)
    print(final.shape)
    np.savez(f'data/split/data_6_split_{int(sheet_name)-1}.npz', data_df=final)
    
