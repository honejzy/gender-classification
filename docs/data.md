### Manifest format for train (.csv)
| audio_path   | label |
| -----------  | ----- |
| path/to/audio.wav   | female   |
| path/to/audio2.wav   | male   |


### Manifest format for inference (.csv)
| ch   | start | end | path | gender | 
| ---  | ---   | --- | ---  | ---    |
| 0   | 1.0   | 5.0 | audio.wav | female
| 1   | 3.5   | 8.2 | audio2.wav | male

column separator is `,`
* **ch** - channel of audio
* **start** - start sec in audio
* **end** - end sec in audio
* **path** - path/to/audio.wav
* **gender** - label of one of two classes ("female", "male")

### Data folders
* **raw** - data with source not processed manifests
* **interim** - data with intermediate stages processing
* **processed** - manifests that are ready for train/test