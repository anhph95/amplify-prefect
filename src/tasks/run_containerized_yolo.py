from prefect import task
import docker


@task()
def run_containerized_yolo(data_dir, output_dir, model_name, epochs, gpus, imgsz, batch, lr0, agnostic_nms, yolo_image):
    """
    Run YOLO training in a Docker container.
    """
    client = docker.from_env()

    volumes = {
        data_dir: {'bind': '/data', 'mode': 'rw'},
        output_dir: {'bind': '/output', 'mode': 'rw'}
    }

    command = f'yolo train data=/data/dataset.yaml model={model_name}.pt epochs={epochs} imgsz={imgsz} batch={batch} lr0={lr0} agnostic_nms={agnostic_nms} project=/output/ device={gpus}'

    container = client.containers.run(
        yolo_image,
        command,
        volumes=volumes,
        device_requests=[docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])],
        ipc_mode="host",
        remove=True,
        detach=False
    )
