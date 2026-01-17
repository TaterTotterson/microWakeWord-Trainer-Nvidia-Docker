# Run training from the command line

## Overview

With these scripts and Dockerfile, you can train new wake words from the
command line without using a Jupyter notebook.

Differences between this Docker image and the Jupyter notebook image:

* The Python training environment isn't included in the image.  Instead, a
  "virtual environment" (venv) is created in the `/data` directory which you
   will have mounted to a host directory. This cuts about 7gb from the image
   and allows the virtualenv to persist across container instances.

* The logic from the Jupyter notebook is contained in individual Python
  and shell scripts

* No ports need to be exposed since the Jupyter notebook server isn't being
  run.

## TL;DR

For the impatient among you...

```shell
$ mkdir /some/work/directory  # On a device with more than 150GB free space
$ docker build -t microwakeword-cli:latest .
$ docker run -it --rm --gpus=all -v /some/work/directory:/data --name=mww-cli microwakeword-cli:latest
root@mww-cli:/# cd /data
root@mww-cli:/data# setup_python_venv
##### You have about 4 minutes to drink coffee

root@mww-cli:/data# setup_training_datasets --cleanup-archives --cleanup-intermediate-files
##### You have about 25 minutes for a quick lunch (on a 1gb/sec internet connection)

root@mww-cli:/data# train_wake_word --cleanup-work-dir "wake_word" "Wake Word"
##### You have about 30-45 minutes for a nap depending on available system resources.
##### You'll be informed of where to find your trained model.
```

Load the trained model on your device and give it a try but don't be surprized
if you get a lot of missed or false activations.  Read on to find out why.

## Get Started

Good, you stuck around!  Now read the rest of the document before doing
anything.

### Using a GPU

Having an Nvidia GPU available can cut the training time by up to half.  The
open-source nouveau driver shipped with Linux kernels doesn't support CUDA
however so if you have an Nvidia GPU and want to use it for training, you'll
need to install the official Nvidia driver from
https://www.nvidia.com/en-in/drivers/unix/

### Build the image

You can use either Docker or Podman as your container management tool.
`docker` is used in the examples but if you have podman, just substitute
the command.

Start by navigating to the directory that contains this README file and
the accompanying Dockerfile.  Then...


```shell
docker build -t microwakeword-cli:latest .
```

This should be fairly quick and result in an image that's about 320mb in size
as it's basically a standard Ubunbtu24.04 image with a few added tools.

So why isn't a pre-built image available for download?  Because it'll probably
take longer to download a pre-built image than for you to create it locally.
GitHub's container registry is notoriously erratic when it comes to download
throughput.

### Create a host work directory

This directory will contain the Python virtual environment plus all of the
downloaded and generated data needed for training and the final trained
models.  A full environment will need about 150gb of free space but read
further to see how to reduce this.

Your `<host_data_dir>` will be mounted inside the container as `/data`.

The training container will start a Bash shell so if you have Bash
aliases or Bashy things you like, create a `.bashrc` file in your
`<host_data_dir>` and put them in there.  It'll automatically be included
any time you enter the container.

### Create and start the container

There are lots of options that control container creation.  The simplest example
will create the container and give you an interactive shell.  When you exit the
shell, the container will be stopped and removed leaving your `<host_data_dir>`
intact.

```shell
$ docker run -it --rm --gpus=all -v <host_work_directory>:/data microwakeword-cli:latest
```

Options:

* Remove the `--gpus=all` option if you don't have an Nvidia GPU or don't want to use it.
* Remove the `--rm` and add a `--name=mww-cli` option to keep the container
  around and give it a name for training more than one wake word.  You
  can stop and remove it when you're ready.
* Add a `-d` option to start the container in the background and use `docker
  attach mww-cli` or `docker exec -it mww-cli /bin/bash` to connect to it.

When the container starts, you'll see:

```text
=======================================================
WARNING: A python virtual environment wasn't found
at /data/.venv.  You'll need to run setup_python_venv
before you'll be able to use this container for
training.
=======================================================
root@mww-cli:/#
```

Don't worry about the python WARNING right now.  You'll be creating the
virtualenv in the next step.

If you've forgotton to create and/or mount your host data directory, you'll
see an additional warning:

```text
=======================================================
WARNING: The /data directory is NOT mounted.
Running the training process without /data mounted
could add over 140Gb of python packages and training
files to this container's storage which is probably
NOT what you want.

You should remove this container and re-create it with
a 'docker run' option like '-v <host_work_dir>:/data'
making sure the host directory is on a device that has
enough free space.
=======================================================
```

You can certainly continue but it's a "really bad idea"™ because your
container storage could grow from a few hundred mb to over 140gb.

At this point, you're in a Bash shell.

### Create the Python virtual environment

The Python virtual environment will contain all the software needed to train.
It gets created as `/data/.venv` and will take up about 11gb of disk space.

The scripts that do all the work will be in the container's PATH so to setup
the virtual environment and install all of the packages, just run:

```text
setup_python_venv [ --verbose ]

Options:

--verbose: Print the detailed "pip install" output.

```

When the installation is finished, a test of the major components will be
run.

Once the process is done, you should change to the `/data` directory and
activate the virtual environment with:

```shell
root@mww-cli:/# cd /data
root@mww-cli:/data# source .venv/bin/activate
(.venv) root@mww-cli:/data#
```

Technically, you don't need to do either of these since the scripts
are in the PATH and they know to use the `/data` directory for everything.
It's more of an "if you're interested" thing.

At this point, you have a container with all software installed.

## Get the reference data

The training process itself relies on a significant amount of audio reference
data that creates a simulated "audio environment" that your wake word will be
trained in.  These "training datasets" include things like varying amounts of
reverberation, background music, background conversations, background noise,
etc.  All said and done, it amounts to about 30gb of audio but with the
downloaded archives and extracted intermediate files, you'll need about 85gb
of free space.  Thankfully, you only need to download the files once no
matter how many wake words you want to train and since it's stored in
`/data`,  you can even remove the docker container and recreate it without
losing any of it.  There are 4 datasets that are required.

This is a three stage process...

1.  Download zipfiles or tarballs.    (about 30gb)
2.  Extract them.                     (about 50gb)
3.  Convert them into the final form. (about 31gb)

NOTE: The sizes add up to more than the 85gb stated earlier because one
of the datasets doesn't need to be covnerted and is counted in both
steps 2 and 3.  You really do only need 85gb.

To download the archives, unpack them, and convert the audio to what's needed
by the training process, run:

```text
setup_training_datasets [ --cleanup-archives ] [ --cleanup-intermediate-files ]

Options:
--cleanup-archives:           Automatically delete the tarballs or zipfiles after
                              they've been extracted.

--cleanup-intermediate-files: Automatically delete the intermediate files
                              after they've been converted.

```

On a 1gb/sec Internet connection, this will take about 25 minutes.

The script detects if the datasets have already been downloaded, extracted
and/or converted and skips those steps as appropriate so if you've run the
script without the cleanup options, you can just run it again with those
options to clean them up.

Now you're ready to train a wake word.  Almost.

## Train a Wake Word

Training is done in 3 stages.

1.  Generate thousands of samples of the wake word with various voices,
pitches, speeds, inflections, etc.
2.  Augment the samples with the training datasets to add background noise, etc.
3.  Run the Tensorflow training.

### Generate a sample for verification

Before you start the full process, you're going to want to generate a single
wake word sample and play it back to ensure it sounds right.  The wake word
should be spelled phonetically to give the sample generator the best chance
of success.

```text
root@mww-cli:/# wake_word_sample_generator --samples=1 "hey buster"
===== Generating 1 sample of 'hey buster' =====
      Loading /data/tools/piper-sample-generator/models/en_US-libritts_r-medium.pt
      Successfully loaded the model
      Batch 1/0 complete
      Done
Sample available at /data/work/test_sample/hey_buster.wav
Play it from your host.
```

You should then play that file from your host.  The reason I used "hey buster"
as the wake word is to demonstrate why it's important to generate and listen
to a sample.  If you try that exact input and play it back, you'll notice
that the generator didn't capture the "er" at the end very well. To get it to
do so, I had to add a period on the end as a "spacer".
"hey buster." worked much better.

When you're happy with the sample, you can run the full process.

### Run the full training process

```text
train_wake_word [ --samples=<samples> ] [ --batch-size=<batch_size> ]
                [ --training-steps=<steps> ] [ --cleanup-work-dir ]
                <wake_word> [ <wake_word_title> ]

Options:
--samples:            The number of samples to generate for the wake word.
                      Default: 20000

--batch-size:         How many samples should be generated at a time.  The more
                      samples, the more memory is needed.
                      Default: 100

--training-steps:     Number of training steps.  More training steps means better
                      detection and false positive rates but also more time to train.
                      Default: 25000

--cleanup-work-dir:   Delete the /data/work directory after successful training.
                      Default: false

<wake_word>           The word to train spelled phonetically.
                      Required.

<wake_word_title>     An optional pretty name to save to the json metadata file.
                      Default: The wake word with individual words capitalized
                               and punctuation removed.

```

By default, the training process creates 20,000 samples of your wake word and
runs 25,000 training steps.  See [Tensorboard Results](#tensorboard-results)
in the [Extra Credit](#extra-credit) section below for
why these are the defaults.  Depending on resources available, this could take
between 30 and 60 minutes.

The resulting tflite model files and logs will be placed in the
`/data/output/<timestamp>-<wake_word>-<samples>-<training-steps>` directory
and will therefore be available from your host in the directory you mapped
`/data` to.  File names will have non-filename-friendly characters in your
wake word changed to underscores to make things easier.  You'll need both the
tflite and json files to load on your device. Exactly how you load them
depends on the device and is beyond the scope of this project.

The only real measure of success is how well the resulting model works
on a real device.  If you encounter too many missed or false activations,
increasing the number of samples would probably improve the results more
than increasing the number of training steps.  See
[Tensorboard Results](#tensorboard-results) in the [Extra Credit](#extra-credit) section below.

The output from the last step is filtered some by the script but still quite
verbose. The full log will be available in the output directory as
`training.log` if you're interested. Intepreting the log is beyond the scope
of this project however.

You can train additional wake words or change the number of samples and
training steps by simply running `train_wake_word` again. No need to repeat
any of the earlier setup steps.  If you change the wake word or the number of
wake word samples, the work directory will be deleted and all 3 steps re-run.
If you only change the number of training steps, the data from the first two
steps is still valid and only the 3rd step is run.

All of the intermediate data is stored in the `/data/work` directory which will
grow to about 17gb with 20,000 wake word samples.  Once the tflite model is
successfully generated and you're happy with the results, you can delete the
`/data/work` directory.

### Training more than one wake word

Once you have a container running, you
can easily train multiple wake words from your host:

```shell
for wp in "hey_alexa" "hey_jenkins" ; do
  docker exec -it mww-cli train_wake_word --cleanup-work-dir "$wp"
done
```

### Training time examples

Training times depend on lots of things.  These are examples only.
Your Mileage May Vary!!!

```text
===============================================================================
                            Training Summary

CPU: Intel(R) Core(TM) i7-6950X CPU @ 3.00GHz (20 cores)  Memory: 64195 mb
GPU: N/A

                 Generate 10000 samples, 100/batch Elapsed time: 0:06:17
                             Augment 10000 samples Elapsed time: 0:04:05
                              10000 training steps Elapsed time: 0:15:04
                              ==================================================
                                             Total Elapsed time: 0:25:26
================================================================================

================================================================================
                            Training Summary

CPU: Intel(R) Core(TM) i7-6950X CPU @ 3.00GHz (20 cores)  Memory: 64195 mb
GPU: NVIDIA GeForce RTX 3060 (3584 cores)  Memory: 11909 mb

                 Generate 10000 samples, 100/batch Elapsed time: 0:00:29
                             Augment 10000 samples Elapsed time: 0:03:40
                              10000 training steps Elapsed time: 0:08:00
                          ======================================================
                                             Total Elapsed time: 0:12:09
================================================================================

================================================================================
                            Training Summary

CPU: Intel(R) Core(TM) i7-6950X CPU @ 3.00GHz (20 cores)  Memory: 64195 mb
GPU: N/A

                 Generate 20000 samples, 100/batch Elapsed time: 0:10:38
                             Augment 20000 samples Elapsed time: 0:07:04
                              25000 training steps Elapsed time: 0:25:21
                          ======================================================
                                             Total Elapsed time: 0:43:03
================================================================================

================================================================================
                            Training Summary

CPU: Intel(R) Core(TM) i7-6950X CPU @ 3.00GHz (20 cores)  Memory: 64195 mb
GPU: NVIDIA GeForce RTX 3060 (3584 cores)  Memory: 11909 mb

                 Generate 20000 samples, 100/batch Elapsed time: 0:00:53
                             Augment 20000 samples Elapsed time: 0:07:05
                              25000 training steps Elapsed time: 0:19:13
                          ======================================================
                                             Total Elapsed time: 0:27:11
================================================================================

================================================================================
                            Training Summary

CPU: Intel(R) Core(TM) i7-6950X CPU @ 3.00GHz (20 cores)  Memory: 64195 mb
GPU: N/A

                 Generate 50000 samples, 100/batch Elapsed time: 0:30:47
                             Augment 50000 samples Elapsed time: 0:20:22
                              40000 training steps Elapsed time: 1:01:51
                              ==================================================
                                             Total Elapsed time: 1:53:00
================================================================================

================================================================================
                            Training Summary

CPU: Intel(R) Core(TM) i7-6950X CPU @ 3.00GHz (20 cores)  Memory: 64195 mb
GPU: NVIDIA GeForce RTX 3060 (3584 cores)  Memory: 11909 mb

                 Generate 50000 samples, 100/batch Elapsed time: 0:02:08
                             Augment 50000 samples Elapsed time: 0:19:13
                              40000 training steps Elapsed time: 0:42:23
                          ======================================================
                                             Total Elapsed time: 1:03:44
================================================================================


```

The sample generation process is really the only one that uses multiple CPUs so
having fewer CPU threads available will probably make little difference.

## Extra Credit

### Training defaults

If you plan on training multiple wake words, you can set your own default
training parameters by creating a `/data/.defaults.env` file with the
following contents:

```shell
# Variable names follow the command line parameters converted to upper case
# and with the dashes ('-') converted to underscores ('_').
export SAMPLES=10000
export TRAINING_STEPS=10000

# Don't use the GPU for any operations.  Stick with the CPU only.
##export CUDA_VISIBLE_DEVICES=-1

```

### Examine your model with Tensorboard

Tensorboard is a web-based graphical model viewer.  You can use it to get an
idea of how many training steps are needed before accuracy results stop
improving.  To use it, you'll have to expose port 6006 by adding `-p
6006:6006` to your `docker run` command line.  If you didn't, don't worry.
Remember, the /data directory is mapped to a directory on your host so you
can simply stop and delete the current container and recreate it with the new
`docker run` command. No need to re-run any of the setup or training steps.

To start Tensorboard, run:

```shell
root@mww-cli:/# cd /data
root@mww-cli:/data# source .venv/bin/activate
(.venv) root@mww-cli:/data# tensorboard --bind_all --logdir ./output
```

Now on your host, point your browser at `http://localhost:6006/`,
click "SCALARS" at the top and take a look at the various charts.  You'll see
a "train" and "validation" item for each training run you've performed.  It's
the "train" items you're interested in.

<a id="tensorboard-results"></a>

You have to be a Tensorflow expert to decipher most of the charts but
the "Accuracy" chart for this particular wake word and 50,000 samples would
seem to idicate that there's very little improvement after about 20,000
training steps.

![Accuracy Chart, 50000 samples](tensorboard1.png)

In contrast, with only 5,000 wake word samples, there's still improvement to be had after
20,000 training steps.

![Accuracy Chart, 5000 samples](tensorboard2.png)

Given that it's faster to generate wake word samples than it is to train,
20,000 samples and 25,000 training steps seems like a good compromise.  This
chart has a bit less smoothing to show a bit more detail and includes the
50,000 sample run as well.  This run took only 27 minutes as opposed to the
63 minutes it took for the 50,000 sample run.  Now you know why 20,000 and
25,000 are the defaults for these scripts.

![Accuracy Chart, 25000 samples](tensorboard3.png)






