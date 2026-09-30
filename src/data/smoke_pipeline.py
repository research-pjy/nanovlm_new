"""Run the existing DATA pipeline in an isolated small-data workspace on rama."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-config',required=True,type=Path)
    parser.add_argument('--num-images',type=int,default=100)
    parser.add_argument('--work-dir',required=True,type=Path)
    parser.add_argument('--short-config',type=Path,default=Path('configs/shortdesc.qwen.json'))
    parser.add_argument('--long-config',type=Path,default=Path('configs/longdesc.qwen.json'))
    parser.add_argument('--validation-config',type=Path,default=Path('configs/description_validation.json'))
    args=parser.parse_args()
    w=args.work_dir
    def call(module,*options):
        print(f'Running {module}',flush=True)
        result=subprocess.run([sys.executable,'-m','src.data.'+module,*map(str,options)])
        # Generation exit 2 means saved invalid outputs; continue to their validation report.
        if result.returncode and not (module.startswith('generate_') and result.returncode==2):
            raise RuntimeError(f'{module} stopped with exit code {result.returncode}')
    try:
        call('prepare_small','--data-config',args.data_config,'--num-images',args.num_images,'--output-dir',w/'inputs')
        for phase,config in [('short',args.short_config),('long',args.long_config)]:
            call('generate_'+phase+'desc','--config',config,'--metadata',w/'inputs/metadata.json',
                 '--splits',w/'inputs/splits.json','--output-dir',w/phase)
            call('validate_descriptions','--input',w/phase/(phase+'desc.jsonl'),'--field',phase+'_desc',
                 '--config',args.validation_config,'--report',w/(phase+'_validation.json'))
        call('export_dataset','--metadata',w/'inputs/metadata.json','--splits',w/'inputs/splits.json',
             '--short-dir',w/'short','--long-dir',w/'long','--output-dir',w/'final')
        call('check_dataset','--dataset',w/'final','--data-config',args.data_config,
             '--metadata',w/'inputs/metadata.json','--splits',w/'inputs/splits.json','--report',w/'dataset_check.json')
    except (OSError,RuntimeError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__': sys.exit(main())
