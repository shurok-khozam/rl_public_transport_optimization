import xml.etree.cElementTree as ET
import subprocess
# OCTAVE_EXEC = 'C:/Program Files/GNU Octave/Octave-7.1.0/mingw64/bin/octave.bat' # Octave 7
OCTAVE_EXEC = 'C:/Program Files/GNU Octave/Octave-6.3.0/mingw64/bin/octave.bat' # Octave 6


class OctaveAdapter:

    def __init__(self, file_context):
        self.file_context = file_context

    def write_to_octave(self, nbr_trains, v_max, max_dwell, density_max_opt):
        # write to: python_to_octave.xml ########
        print("------> Starting creating python_to_octave file")
        parameters_element = ET.Element("parameters")
        parameters_element.text = "\n\t"

        nbr_trains_element = ET.SubElement(parameters_element, "nbr_trains")
        nbr_trains_element.text = str(nbr_trains)
        nbr_trains_element.tail = "\n\t"

        v_max_element = ET.SubElement(parameters_element, "v_max")
        v_max_element.text = str(v_max)
        v_max_element.tail = "\n\t"

        max_dwell_element = ET.SubElement(parameters_element, "max_dwell")
        max_dwell_element.text = str(max_dwell)
        max_dwell_element.tail = "\n\t"

        density_max_opt_element = ET.SubElement(parameters_element, "density_max_opt")
        density_max_opt_element.text = str(density_max_opt)
        density_max_opt_element.tail = "\n"

        python_to_octave = ET.ElementTree(parameters_element)
        python_to_octave.write(self.file_context + "/io/python_to_octave.xml")
        print("<------ Terminated creating python_to_octave file")


    def run_octave(self):
        print("------> Starting Octave code")
        process = subprocess.Popen([OCTAVE_EXEC, '--persist', 'main.m'], cwd=self.file_context)
        stdout, stderr = process.communicate()
        process.wait()
        print("<------ Terminated Octave code")


    def read_from_octave(self):
        # read from: octave_to_python.xml #######
        print("------> Starting reading octave_to_python file")
        octave_to_python = ET.parse(self.file_context + '/io/octave_to_python.xml')
        octave_to_python_params = octave_to_python.getroot()
        out = {
            'h_out_mean': float(octave_to_python_params.find('h_out_mean').text),
            'h_out_std': float(octave_to_python_params.find('h_out_std').text),
            'I_mean': float(octave_to_python_params.find('I_mean').text),
            'I_std': float(octave_to_python_params.find('I_std').text),
            'A_mean': float(octave_to_python_params.find('A_mean').text),
            'A_std': float(octave_to_python_params.find('A_std').text),
            'mu_mean': float(octave_to_python_params.find('mu_mean').text),
            'mu_std': float(octave_to_python_params.find('mu_std').text),
            'Q_mean': float(octave_to_python_params.find('Q_mean').text),
            'Q_std': float(octave_to_python_params.find('Q_std').text),
            'P_mean': float(octave_to_python_params.find('P_mean').text),
            'P_std': float(octave_to_python_params.find('P_std').text),
            'dwell_mean': float(octave_to_python_params.find('dwell_mean').text),
            'dwell_std': float(octave_to_python_params.find('dwell_std').text)
        }
        print("<------ Terminated reading octave_to_python file")
        return out