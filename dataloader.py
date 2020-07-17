from torch.utils.data import Dataset

class ISLES2018Dataset_MTT(Dataset):
    def __init__(self, folder, modalities=None):
        self.samples = []
        
        for case_name in os.listdir(folder):
            case_path = os.path.join(folder, case_name)
            case = {}
            for file_path in os.listdir(case_path):
                modality = re.search(r'XX.O.(\w+).\d+',file_path).group(1)
                if modality != 'CT_4DPWI': #change to modality in modalities but das slow
                    nii_path_name = os.path.join(case_path,file_path,file_path+'.nii')
                    img = nib.load(nii_path_name)
                    case[modality] = img
            
            for i in range(case['CT'].shape[2]):
                arr = []
                for modality in modalities:
                    if modality != 'OT':
                        arr.append(torch.from_numpy(case[modality].get_fdata()[:,:,i]).float().unsqueeze(0))
                new = torch.cat(tuple(arr), dim=0)
                self.samples.append((new, torch.from_numpy(case['OT'].get_fdata()[:,:,i]).float().unsqueeze(0) ))
                
        print("Cases loaded: ", str(len(self.samples)))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
