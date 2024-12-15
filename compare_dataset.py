import os

def get_dataset_structure(root_dir):
    """
    주어진 루트 디렉토리에서 데이터셋 구조를 반환합니다.
    """
    dataset_structure = {}
    speed_path = os.path.join(root_dir, "speed_30")
    if not os.path.exists(speed_path):
        raise ValueError(f"Directory {speed_path} does not exist.")
    
    for angle_folder in sorted(os.listdir(speed_path)):
        angle_path = os.path.join(speed_path, angle_folder)
        if not os.path.isdir(angle_path):
            continue
        
        images = sorted(os.listdir(angle_path))
        dataset_structure[angle_folder] = set(images)
    
    return dataset_structure

def compare_datasets(root_dir1, root_dir2):
    """
    두 데이터셋의 구조를 비교하여 차이점을 출력합니다.
    """
    dataset1 = get_dataset_structure(root_dir1)
    dataset2 = get_dataset_structure(root_dir2)
    
    all_angle_folders = set(dataset1.keys()).union(set(dataset2.keys()))
    differences = {}

    for angle_folder in all_angle_folders:
        images1 = dataset1.get(angle_folder, set())
        images2 = dataset2.get(angle_folder, set())
        
        only_in_dataset1 = images1 - images2
        only_in_dataset2 = images2 - images1
        
        if only_in_dataset1 or only_in_dataset2:
            differences[angle_folder] = {
                "only_in_dataset1": only_in_dataset1,
                "only_in_dataset2": only_in_dataset2
            }
    
    return differences

def print_sorted_differences(differences):
    """
    차이점을 정렬하여 출력합니다.
    """
    if not differences:
        print("두 데이터셋은 동일합니다.")
        return

    for angle_folder, diff in sorted(differences.items()):
        print(f"\n[폴더]: {angle_folder}")

        # Dataset1에만 있는 이미지 출력
        if diff["only_in_dataset1"]:
            sorted_only_in_dataset1 = sorted(diff["only_in_dataset1"])
            print(f"  - Dataset1에만 있는 이미지 ({len(sorted_only_in_dataset1)}개):")
            for img in sorted_only_in_dataset1:
                print(f"    - {img}")

        # Dataset2에만 있는 이미지 출력
        if diff["only_in_dataset2"]:
            sorted_only_in_dataset2 = sorted(diff["only_in_dataset2"])
            print(f"  - Dataset2에만 있는 이미지 ({len(sorted_only_in_dataset2)}개):")
            for img in sorted_only_in_dataset2:
                print(f"    - {img}")

# 사용 예시
root_dir1 = "/Users/kyumin/AutoDriving/artifacts/speed30_and_100_60x60:v2"
root_dir2 = "/Users/kyumin/AutoDriving/artifacts/speed30_and_100_60x60:v3"

differences = compare_datasets(root_dir1, root_dir2)
print_sorted_differences(differences)
