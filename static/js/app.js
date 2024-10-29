// 이미지 미리보기 함수
function previewImage(input, previewId) {
    const file = input.files[0];
    const preview = document.getElementById(previewId);

    if (file) {
        const reader = new FileReader();

        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = 'block';
        }

        reader.readAsDataURL(file);
    } else {
        preview.src = '#';
        preview.style.display = 'none';
    }
}

// 파일 입력 필드에 이벤트 리스너 추가
document.getElementById('face').addEventListener('change', function() {
    previewImage(this, 'facePreview');
});

document.getElementById('body').addEventListener('change', function() {
    previewImage(this, 'bodyPreview');
});

document.getElementById('body_handsup').addEventListener('change', function() {
    previewImage(this, 'bodyHandsupPreview');
});

// 추가된 변수 및 함수 선언
let currentPage = 0;
let isFinalPage = false; // 마지막 페이지 여부
let isLoading = false;
let userAttributes = {}; // 사용자 속성을 저장할 객체
let selectedSeason = ''; // 선택된 계절
let manualInput = false; // 신체 정보 직접 입력 여부

// 모달 관련 변수
let ratingModal = document.getElementById('ratingModal');
let ratingImage = document.getElementById('ratingImage');
let ratingStars = document.querySelectorAll('.rating-stars .star');
let submitRatingButton = document.getElementById('submitRatingButton');
let selectedRating = 0;
let currentRatedItem = '';

// 성별 선택 시 체형 옵션 업데이트 함수
function updateBodyShapeOptions() {
    const gender = document.getElementById('gender').value;
    const bodyshapeSelect = document.getElementById('bodyshape');

    // 기존 옵션 모두 제거
    while (bodyshapeSelect.firstChild) {
        bodyshapeSelect.removeChild(bodyshapeSelect.firstChild);
    }

    if (!gender) {
        // 성별이 선택되지 않은 경우
        bodyshapeSelect.disabled = true;
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = '성별을 먼저 선택하세요';
        bodyshapeSelect.appendChild(defaultOption);
        return;
    } else {
        bodyshapeSelect.disabled = false;
    }

    // 기본 옵션 추가
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = '선택하세요';
    bodyshapeSelect.appendChild(defaultOption);

    // 공통 옵션
    const commonOptions = [
        { value: 'round', text: '원형' },
        { value: 'rectangle', text: '직사각형' },
        { value: 'inverted_triangle', text: '역삼각형' },
        { value: 'triangle', text: '삼각형' }
    ];

    // 성별에 따른 옵션 추가
    if (gender === 'man') {
        // 남성용 옵션
        const maleOptions = [
            { value: 'trapezoid', text: '사다리꼴형' }
        ];
        [...maleOptions, ...commonOptions].forEach(optionData => {
            const option = document.createElement('option');
            option.value = optionData.value;
            option.textContent = optionData.text;
            bodyshapeSelect.appendChild(option);
        });
    } else if (gender === 'woman') {
        // 여성용 옵션
        const femaleOptions = [
            { value: 'hourglass', text: '모래시계형' }
        ];
        [...femaleOptions, ...commonOptions].forEach(optionData => {
            const option = document.createElement('option');
            option.value = optionData.value;
            option.textContent = optionData.text;
            bodyshapeSelect.appendChild(option);
        });
    }
}

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', function() {
    updateBodyShapeOptions(); // 초기 옵션 설정
});

// 성별 선택 변경 시 이벤트 리스너 추가
document.getElementById('gender').addEventListener('change', function() {
    updateBodyShapeOptions();
});

document.getElementById('uploadForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    // 이전 결과 및 오류 메시지 초기화
    document.getElementById('recommendations').innerHTML = '';
    document.getElementById('error').innerText = '';
    document.getElementById('loadMoreContainer').style.display = 'none';
    currentPage = 0;
    isFinalPage = false;

    // 기존 personal-info 제거
    const existingPersonalInfoDivs = document.querySelectorAll('.personal-info');
    existingPersonalInfoDivs.forEach(div => div.parentNode.removeChild(div));

    // 성별과 나이 가져오기
    const gender = document.getElementById('gender').value;
    const age = document.getElementById('age').value;

    // 성별 선택 확인
    if (!gender) {
        document.getElementById('error').innerText = '성별을 선택하세요.';
        return;
    }

    // 표시 로딩 상태
    document.getElementById('loading').style.display = 'block';

    // 계절 선택 여부 확인
    selectedSeason = document.getElementById('season').value;

    try {
        if (manualInput) {
            // 신체 정보 직접 입력한 경우
            const color = document.getElementById('color').value;
            const faceshape = document.getElementById('faceshape').value;
            const bodyshape = document.getElementById('bodyshape').value;

            if (!color || !faceshape || !bodyshape) {
                throw new Error('퍼스널 컬러, 얼굴형, 체형을 모두 선택하세요.');
            }

            // 사용자 속성 설정
            userAttributes = { gender, age, color, faceshape, bodyshape };

            // 개인 정보 표시
            displayPersonalInfo(color, faceshape, bodyshape);

            // 첫 번째 추천 요청
            await fetchAndDisplayRecommendations(currentPage);
        } else {
            // 이미지 업로드 및 사용자 속성 획득
            const formData = new FormData();
            formData.append('face', document.getElementById('face').files[0]);
            formData.append('body', document.getElementById('body').files[0]);
            formData.append('body_handsup', document.getElementById('body_handsup').files[0]);
            formData.append('gender', gender);
            formData.append('age', age);

            if (selectedSeason) {
                formData.append('season', selectedSeason);
            }

            // 이미지 업로드 필드 검증
            if (!document.getElementById('face').files[0] || !document.getElementById('body').files[0] || !document.getElementById('body_handsup').files[0]) {
                throw new Error('모든 이미지를 업로드하세요.');
            }

            // 1단계: 이미지 업로드 및 사용자 속성 획득
            const uploadResponse = await fetch('http://localhost:5000/upload', {
                method: 'POST',
                body: formData
            });

            const uploadResult = await uploadResponse.json();

            if (!uploadResponse.ok) {
                throw new Error(uploadResult.error || '이미지 처리에 실패했습니다.');
            }

            // 사용자 속성 추출
            const { color, faceshape, bodyshape } = uploadResult;
            userAttributes = { gender, age, color, faceshape, bodyshape };

            // 개인 정보 표시
            displayPersonalInfo(color, faceshape, bodyshape);

            // 첫 번째 추천 요청
            await fetchAndDisplayRecommendations(currentPage);
        }
    } catch (error) {
        document.getElementById('error').innerText = error.message;
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
});

document.getElementById('loadMoreButton').addEventListener('click', async function() {
    if (isLoading || isFinalPage) return;
    currentPage += 1;
    await fetchAndDisplayRecommendations(currentPage);
});

// 신체 정보 직접 입력하기 버튼 이벤트 리스너
document.getElementById('toggleInputMethodButton').addEventListener('click', function() {
    manualInput = !manualInput;
    const photoSection = document.getElementById('photoUploadSection');
    const manualSection = document.getElementById('manualInputSection');
    const button = document.getElementById('toggleInputMethodButton');

    if (manualInput) {
        // 신체 정보 직접 입력 활성화
        photoSection.style.display = 'none';
        manualSection.style.display = 'block';
        button.innerText = '사진으로 분석하기';
    } else {
        // 사진 업로드로 전환
        photoSection.style.display = 'block';
        manualSection.style.display = 'none';
        button.innerText = '신체 정보 직접 입력하기';
    }
});

async function fetchAndDisplayRecommendations(page) {
    isLoading = true;
    // 표시 로딩 상태
    document.getElementById('loading').style.display = 'block';
    document.getElementById('error').innerText = '';

    try {
        // 추천 요청
        let recommendEndpoint = 'http://localhost:5000/recommend';
        const requestBody = {
            ...userAttributes,
            page: page
        };

        if (selectedSeason) {
            recommendEndpoint = 'http://localhost:5000/recommend_season';
            requestBody.season = selectedSeason;
        }

        const recommendResponse = await fetch(recommendEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        const recommendResult = await recommendResponse.json();

        if (!recommendResponse.ok) {
            throw new Error(recommendResult.error || '추천을 받는 데 실패했습니다.');
        }

        // 추천 결과 표시
        const hasMore = await displayRecommendations(recommendResult, userAttributes.gender);

        // 'isfinal' 헤더 확인
        if (selectedSeason) {
            const isFinalHeader = recommendResponse.headers.get('isfinal');
            isFinalPage = isFinalHeader === '1';
        } else {
            // 기존 /recommend 엔드포인트에서는 'isfinal' 헤더가 없으므로, 아이템 수로 판단
            isFinalPage = recommendResult && Object.keys(recommendResult).length < 5;
        }

        // "더 보기" 버튼 표시 여부 결정
        if (!isFinalPage) {
            document.getElementById('loadMoreContainer').style.display = 'block';
        } else {
            document.getElementById('loadMoreContainer').style.display = 'none';
        }
    } catch (error) {
        document.getElementById('error').innerText = error.message;
    } finally {
        document.getElementById('loading').style.display = 'none';
        isLoading = false;
    }
}

async function displayRecommendations(data, gender) {
    const recommendationsDiv = document.getElementById('recommendations');

    // 제목 추가 (처음 페이지에만)
    if (currentPage === 0) {
        const title = document.createElement('h2');
        title.innerText = '추천 패션 아이템';
        recommendationsDiv.appendChild(title);
    }

    const items = Object.values(data);

    if (items.length === 0) {
        if (currentPage === 0) {
            recommendationsDiv.innerHTML += '<p>추천된 패션 아이템이 없습니다.</p>';
        }
        return false; // 더 이상 페이지가 없음
    }

    for (const item of items) {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'recommendation-item';

        // /photo 엔드포인트를 통해 이미지 URL을 가져옴
        try {
            const photoResponse = await fetch(`http://localhost:5000/photo/${encodeURIComponent(gender)}/${encodeURIComponent(item.image)}`);
            const photoResult = await photoResponse.json();

            if (!photoResponse.ok) {
                throw new Error(photoResult.error || '이미지 URL을 가져오는 데 실패했습니다.');
            }

            const imageUrl = photoResult; // CloudFront 이미지 URL

            const image = document.createElement('img');
            image.src = imageUrl;
            image.alt = item.image;

            // 이미지 클릭 이벤트 리스너 추가 (평점 모달 열기)
            image.addEventListener('click', function() {
                openRatingModal(item.image, imageUrl);
            });

            const info = document.createElement('p');
            info.innerHTML = `<strong>예상 평점:</strong> ${item.predict.toFixed(2)}<br>
                              <strong>평균 평점:</strong> ${item.average.toFixed(2)}<br>
                              <strong>총 점수:</strong> ${item.total.toFixed(2)}`;

            itemDiv.appendChild(image);
            itemDiv.appendChild(info);

            // 옷 정보 API 호출
            try {
                const clothesInfoResponse = await fetch(`http://localhost:5000/info/${encodeURIComponent(gender)}/${encodeURIComponent(item.image)}`);
                const clothesInfoResult = await clothesInfoResponse.json();

                if (!clothesInfoResponse.ok) {
                    throw new Error(clothesInfoResult.error || '옷 정보 가져오기에 실패했습니다.');
                }

                // 옷 정보 추출 (첫 번째 키의 값 사용)
                const clothesInfo = Object.values(clothesInfoResult)[0];

                // 옷의 비-불리언 필드 및 특징 추출
                const nonBooleanFields = {};
                const features = [];

                for (const [key, value] of Object.entries(clothesInfo)) {
                    if (key === 'image' || key === 'item_gender') {
                        // 'image'와 'item_gender'는 표시하지 않음
                        continue;
                    }
                    if (key === 'style') {
                        nonBooleanFields['스타일'] = value; // 'style'을 '스타일'로 표시
                    } else if (typeof value === 'boolean') {
                        if (value) {
                            features.push(key);
                        }
                    } else {
                        nonBooleanFields[key] = value;
                    }
                }

                // 비-불리언 필드 표시
                const clothesInfoDiv = document.createElement('div');
                clothesInfoDiv.className = 'clothes-info';

                for (const [key, value] of Object.entries(nonBooleanFields)) {
                    const paragraph = document.createElement('p');
                    paragraph.innerHTML = `<strong>${key}:</strong> ${value}`;
                    clothesInfoDiv.appendChild(paragraph);
                }

                itemDiv.appendChild(clothesInfoDiv);

                // 특징 섹션 추가 (true인 항목만)
                if (features.length > 0) {
                    const featuresDiv = document.createElement('div');
                    featuresDiv.className = 'features';

                    const featuresParagraph = document.createElement('p');
                    featuresParagraph.innerHTML = `<strong>특징:</strong> ${features.join(', ')}`;
                    featuresDiv.appendChild(featuresParagraph);

                    itemDiv.appendChild(featuresDiv);
                }
            } catch (err) {
                // 옷 정보 로딩 실패 시 대체 텍스트 표시
                const errorDiv = document.createElement('div');
                errorDiv.className = 'error';
                errorDiv.innerText = `옷 정보 로딩 실패: ${item.image}`;
                itemDiv.appendChild(errorDiv);
            }

            recommendationsDiv.appendChild(itemDiv);
        } catch (err) {
            // 이미지 로딩 실패 시 대체 텍스트 표시
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.innerText = `이미지 로딩 실패: ${item.image}`;
            recommendationsDiv.appendChild(errorDiv);
        }
    }

    // 추가 페이지가 있는지 판단
    if (items.length < 5) { // 페이지당 5개 아이템 가정
        return false; // 더 이상 페이지가 없음
    }

    return true; // 더 보기 가능
}

function displayPersonalInfo(color, faceshape, bodyshape) {
    const recommendationsDiv = document.getElementById('recommendations');
    const parentDiv = recommendationsDiv.parentNode;

    // 기존 personal-info 제거
    let personalInfoDiv = parentDiv.querySelector('.personal-info');
    if (personalInfoDiv) {
        parentDiv.removeChild(personalInfoDiv);
    }

    personalInfoDiv = document.createElement('div');
    personalInfoDiv.className = 'personal-info';

    const infoTitle = document.createElement('h3');
    infoTitle.innerText = '신체적 특징 분석 결과';
    personalInfoDiv.appendChild(infoTitle);

    const colorPara = document.createElement('p');
    colorPara.innerHTML = `<strong>퍼스널 컬러:</strong> ${color}`;
    personalInfoDiv.appendChild(colorPara);

    const faceshapePara = document.createElement('p');
    faceshapePara.innerHTML = `<strong>얼굴형:</strong> ${faceshape}`;
    personalInfoDiv.appendChild(faceshapePara);

    const bodyshapePara = document.createElement('p');
    bodyshapePara.innerHTML = `<strong>체형:</strong> ${bodyshape}`;
    personalInfoDiv.appendChild(bodyshapePara);

    // personalInfoDiv를 recommendationsDiv 이전에 삽입
    parentDiv.insertBefore(personalInfoDiv, recommendationsDiv);
}

// 모달 열기 함수
function openRatingModal(imageName, imageUrl) {
    currentRatedItem = imageName;
    ratingImage.src = imageUrl;
    ratingModal.style.display = 'block';
}

// 별점 클릭 이벤트 리스너 추가
ratingStars.forEach(function(star) {
    star.addEventListener('click', function() {
        selectedRating = this.getAttribute('data-value');
        // 선택된 별점 표시 업데이트
        ratingStars.forEach(function(s) {
            if (s.getAttribute('data-value') <= selectedRating) {
                s.classList.add('selected');
            } else {
                s.classList.remove('selected');
            }
        });
    });
});

// 평점 제출 버튼 클릭 이벤트
submitRatingButton.addEventListener('click', function() {
    if (selectedRating > 0) {
        submitRating();
    } else {
        alert('평점을 선택하세요.');
    }
});

// 평점 제출 함수
function submitRating() {
    // 필요한 데이터 수집
    const data = {
        gender: userAttributes.gender,
        age: userAttributes.age,
        color: userAttributes.color,
        faceshape: userAttributes.faceshape,
        bodyshape: userAttributes.bodyshape,
        clothes: currentRatedItem,
        rating: selectedRating
    };

    // 서버로 데이터 전송
    fetch('http://localhost:5000/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    }).then(function(response) {
        return response.json();
    }).then(function(result) {
        if (result.error) {
            alert('평점 제출에 실패했습니다: ' + result.error);
        } else {
            alert('평점이 제출되었습니다.');
            // 모달 닫기
            ratingModal.style.display = 'none';
            // 선택된 평점 초기화
            selectedRating = 0;
            ratingStars.forEach(function(s) {
                s.classList.remove('selected');
            });
        }
    }).catch(function(error) {
        alert('평점 제출에 실패했습니다.');
    });
}

// 모달 닫기 버튼 이벤트 리스너
document.querySelector('#ratingModal .close').addEventListener('click', function() {
    ratingModal.style.display = 'none';
    // 선택된 평점 초기화
    selectedRating = 0;
    ratingStars.forEach(function(s) {
        s.classList.remove('selected');
    });
});

// 모달 외부 클릭 시 모달 닫기
window.addEventListener('click', function(event) {
    if (event.target == ratingModal) {
        ratingModal.style.display = 'none';
        // 선택된 평점 초기화
        selectedRating = 0;
        ratingStars.forEach(function(s) {
            s.classList.remove('selected');
        });
    }
});
