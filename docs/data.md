### Manifest.csv format
| ch   | start | end | path | gender | 
| ---  | ---   | --- | ---  | ---    |
| 0   | 1.0   | 5.0 | audio.wav | 0
| 1   | 3.5   | 8.2 | audio.wav | 1

column separator is `,`
* **ch** - channel of audio
* **start** - start sec in audio
* **end** - end sec in audio
* **path** - path/to/audio.wav
* **gender** - label of one of two classes ("female" = 0, "male" = 1)

### Data folders
* **raw** - data with source not processed manifests
* **interim** - data with intermediate stages processing
* **processed** - manifests that are ready for train/test