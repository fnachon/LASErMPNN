from pathlib import Path
import shutil
import subprocess


def test_inference():
    this_file_path = Path(__file__)
    laser_dir = this_file_path.parent.parent.absolute()
    installation_dir = laser_dir.parent.absolute()

    example_pdb_path = laser_dir / 'example_pdbs' / '4jnj-1_prot.pdb'

    test_dump_output_path = Path('./pytest_debug/').absolute()
    shutil.rmtree(test_dump_output_path, ignore_errors=True)

    n_outputs = 5

    command = f'cd {installation_dir}; python -m {laser_dir.stem}.run_batch_inference {example_pdb_path} {test_dump_output_path} {n_outputs}'
    print("running inference with the command:\n\t ", command)
    subprocess.run(command, shell=True)

    num_pdbs = len([output for output in test_dump_output_path.glob('*.pdb')])
    assert num_pdbs == n_outputs
    print('The expected number of outputs were created!')
    shutil.rmtree(test_dump_output_path, ignore_errors=True)

