import cv2 as cv


def main():
    # ファイルを読み込み
    image_file = 'input/target.png'
    src = cv.imread(image_file, cv.IMREAD_COLOR)
    # 画像の大きさ取得
    height, width, channels = src.shape
    image_size = height * width
    # グレースケール化
    img_gray = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    # しきい値指定によるフィルタリング
    retval, dst = cv.threshold(img_gray, 127, 255, cv.THRESH_TOZERO_INV )
    # 白黒の反転
    dst = cv.bitwise_not(dst)
    # 再度フィルタリング
    retval, dst = cv.threshold(dst, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # 輪郭を抽出
    contours, hierarchy = cv.findContours(dst, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # この時点での状態をデバッグ出力
    dst = cv.imread(image_file, cv.IMREAD_COLOR)
    dst = cv.drawContours(dst, contours, -1, (0, 0, 255, 255), 2, cv.LINE_AA)
    cv.imwrite('output/debug.png', dst)
    dst = cv.imread(image_file, cv.IMREAD_COLOR)
    for i, contour in enumerate(contours):

        # # 小さな領域の場合は間引く
        area = cv.contourArea(contour)
        if area < 500:
            continue

        # 画像全体を占める領域は除外する
        if image_size * 0.99 < area:
            continue

        # 外接矩形を取得
        x,y,w,h = cv.boundingRect(contour)

        # 縦横比を取得
        if w / h > 0.9 and w / h < 1.1:
          dst = cv.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)

    # 結果を保存
    cv.imwrite('output/result.png', dst)

if __name__ == '__main__':
    main()
