for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done
